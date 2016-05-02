/*
 *  DecisionTree.cpp
 *
 *  Decision Tree class: makes tree based on training data and classifies
 *      test data.
 *
 *  Current Assumptions/Methods:
 *      (1) integer label values
 *      (2) binary splits
 *      (3) for given integer threshold value t: split [ , t) & [t, ]
 *      (4) assumes all feature values are integers
 *      (5) stops when data in each node has same label or min node size reached
 *
 *  Created by Kelsey Schuster
 *  11/3/15
 */


#include "DecisionTree.h"

using namespace std;


namespace trees
{
    DecisionTree::Node::Node()
    {
    }
    
    DecisionTree::Node::~Node()
    {
    }
    
    DecisionTree::Node::Node(int n)
    {
        isLeaf = false;
    }
    
    DecisionTree::Node::Node(const Node& n)
    {
        spltRule = n.spltRule;
        chld = n.chld;
        isLeaf = n.isLeaf;
        lab = n.lab;
    }
    
    const DecisionTree::Node& DecisionTree::Node::operator=(const Node& n)
    {
        spltRule = n.spltRule;
        chld = n.chld;
        isLeaf = n.isLeaf;
        lab = n.lab;
        return *this;
    }
    
    //initialize model and set features
    DecisionTree::DecisionTree(vector<string>& features)
    : _features(features)
    {
        //set number of features
        _nFeatures = _features.size();
        
        //sets # of features to use at each node (use all for basic tree)
        _nConsideredFeatures = _nFeatures;
        
        //get map with integer index of feature string (_featureMap)
        makeFeatureIndexMap(features);
        
        //false is default (won't print out as much info)
        _vocal = false;
        
        //don't follow any samples when making predictions
        _followSampleIndex = -1;
    }
    
    DecisionTree::~DecisionTree()
    {
        cerr << "finished" << endl;
    }
    
    //set whether program prints out splitting features, other info
    void DecisionTree::setVocal(bool v)
    {
        _vocal = v;
    }
    
    //sets the sample whose splits we record when making predictions
    void DecisionTree::followSample(int s)
    {
        _followSampleIndex = s;
    }
    
    //train decision tree with input training data
    void DecisionTree::trainDecisionTree(Matrix& trainData, std::vector<int>& trainLabels, int minSize) 
    {
        //set min # of samples each node can contain
        _minNodeSize = minSize;
        
        //check to make sure correct # of features
        if (trainData.size() == 0 || trainData[0].size() != _nFeatures) {
            cerr << "Error: incorrect number of features" << endl;
            exit(1);
        }
        
        //get possible values for sample label (assumes discrete values)
        _labelValues = getLabelValues(trainLabels);
        
        //set default label by finding most probable (assume same distr)
        _defaultLabel = getLabelMode(trainLabels);
        
        //get possible values for each feature
        _featureValues = getFeatureValues(trainData);
        
        //build classification tree
        Node* root = (Node*) new DecisionTree::Node(2);
        buildDecisionTree(trainData, trainLabels, root);
        
        //save classifier for future use
        _root = root;
    }
    
    
    //given data + labels, param num + values to test, performs k-fold CV
    map<int, double> DecisionTree::performCrossValidation(Matrix& data, vector<int>& labels, vector<int>& paramVals, int param, int k)
    {
        // param = 1: minNodeSize
        
        double invK = 1.0/((double)k);
        Matrix validData, trainData;
        vector<int> validLabels, trainLabels;
        
        //randomize data and labels
        randomizeSamples(data, labels);
        
        //data structure for param values with corresp CV accuracy
        map<int, double> accuracyStore;
        
        //iterate through parameter values
        for (unsigned int i=0; i<paramVals.size(); i++) {
            
            if (_vocal) cout << "parameter value " << paramVals[i] << endl;
            
            //iterate through 1:k (k-fold cross-validation)
            for (unsigned int j=0; j<k; j++) {
                
                if (_vocal) cout << "\tk: " << j << endl;
                
                //pull out validation data and training data
                for (unsigned int n=0; n<data.size(); n++) {
                    if ((n >= (int)(j*data.size()*invK)) && (n < (int)((j+1)*data.size()*invK))) {
                        validData.push_back(data[n]);
                        validLabels.push_back(labels[n]);
                    } else {
                        trainData.push_back(data[n]);
                        trainLabels.push_back(labels[n]);
                    }
                }
                
                //train tree with selected training data
                trainDecisionTree(trainData, trainLabels, paramVals[i]);
                
                //make predictions for selected validation set
                vector<int> predictions(validData.size(), -1);
                makePredictions(validData, predictions);
                
                //compute and store validation accuracy
                accuracyStore[paramVals[i]] += invK*computeValidationAccuracy(predictions, validLabels);
                
                //clear data structures
                validData.clear();
                validLabels.clear();
                trainData.clear(); 
                trainLabels.clear();
            }
        }
        return accuracyStore;
    }
     
    
    //given test data and current decision tree (_root), make label predictions
    void DecisionTree::makePredictions(Matrix& testData, vector<int>& predictions)
    {
        _sampleStore.clear();
        
        //make prediction for each sample in data matrix
        for (unsigned int i=0; i<testData.size(); i++) {
            
            //set prediction to default value to start
            int prediction = _defaultLabel;    
            
            Node* root = _root;
            
            //go down tree until we end up at leaf node
            while (!root->isLeaf) {
                
                //get splitting feature index and corresp threshold for this node
                int featIndex = root->spltRule.first;
                int threshold = root->spltRule.second;
                
                //see which child this data point goes to
                if (testData[i][featIndex] < threshold) {
                    
                    //if set, save split info
                    if (i == _followSampleIndex) {
                        saveSplitInfo(i, featIndex, threshold, 0);
                    }
                    root = root->chld[0];
                    
                } else {
                    
                    //if set, save split info
                    if (i == _followSampleIndex) {
                        saveSplitInfo(i, featIndex, threshold, 1);
                    }
                    root = root->chld[1];
                }
                
                if (root == NULL) {
                    prediction = _defaultLabel;        
                    break;
                }
                
                //make prediction based on label of node
                prediction = root->lab;         
            }
            
            //store prediction for sample
            predictions[i] = prediction;
        }
    }
    
    //if set, save split info for the selected sample
    void DecisionTree::saveSplitInfo(int i, int f, int t, int n)
    {
        DecisionTree::SplitPair p;
        vector<int> v;
        
        //put info in pair structure
        p.first = _features[f];
        v.push_back(t);
        v.push_back(n);
        p.second = v;
        
        //store feature, threshold, and left/right child node info for sample i
        _sampleStore[i].push_back(p);
        
        //clear data structures
        v.clear();
    }
    
    //returns saved split info for selected data point index s
    DecisionTree::SplitVector DecisionTree::getSampleSplits(int s)
    {
        return _sampleStore[s];
    }
    
    //given label prediction vector and actual labels, computes validation accuracy
    double DecisionTree::computeValidationAccuracy(vector<int>& predictions, vector<int>& labels)
    {
        int pSize = predictions.size();
        int accurate = 0;
        
        //check if same number of samples
        if (pSize != labels.size()) {
            cerr << "Error: need same number of samples in predictions and labels" << endl;
            return -1.0;
            
        //then proceed with accuracy rate calculation
        } else {
            for (unsigned int i=0; i<pSize; i++) {
                if (predictions[i] == labels[i]) {
                    accurate++;
                }
            }
        }
        return ((double)accurate)/((double)pSize);
    }
    
    //recursive function to construct decision tree
    DecisionTree::Node* DecisionTree::buildDecisionTree(Matrix& trainData, vector<int>& trainLabels, Node* n)
    {
        
        n->lab = getLabelMode(trainLabels);
        
        //return NULL if table has no data in it
        if (trainData.size() == 0) {
            return NULL;
        }
        
        //if all samples in the node have the same label, stop and label node
        if (sameLabels(trainLabels)) {
            
            //current node is a leaf node, assign label
            n->isLeaf = true;
            n->lab = trainLabels[0];
            return n;
            
        //decide splitting column and branch off
        } else {
            
            //use split feature/threshold rule with maximum info gain
            n->spltRule = findSegmentor(trainData, trainLabels);
            
            //stores subsets of data in each child node (2 b/c binary)
            vector< Matrix > tData(2, Matrix(0));
            vector< vector<int> > tLabels(2, vector<int>(0));
            
            //get subset of data in each child node, check for min node size
            for (unsigned int i=0; i<2; i++) {
                
                trimMatrix(trainData, trainLabels, tData[i], tLabels[i], n->spltRule, i);
                
                //if reached minimum node size, stop and assign leaf node
                if (tData[i].size() < _minNodeSize) {
                    n->isLeaf = true;
                    n->lab = getLabelMode(trainLabels);
                    return n;
                }
            }
            
            //if both proposed children have enough samples in each, proceed
            if (_vocal) {
                cout << "feature: " << _features[n->spltRule.first] << "\tthreshold: " << n->spltRule.second << endl;
                cout << "\tsize left node: " << tData[0].size() << "\tsize right node: " << tData[1].size() << endl;
            }
            
            //make new child nodes and recursively build tree
            for (unsigned int i=0; i<2; i++) {
                
                Node* n2 = (Node*) new DecisionTree::Node(2);
                n2->lab = _defaultLabel;
                
                n->chld.push_back(buildDecisionTree(tData[i], tLabels[i], n2));
            }
        }
        return n;
    }
    
    //trims table to include only data that belongs in the node
    void DecisionTree::trimMatrix(Matrix& data, vector<int>& labels, Matrix& tData,
                                  vector<int>& tLabels, pair<int, int>& seg, int nodeIndex)
    {
        //iterate through samples
        for (unsigned int i=0; i<data.size(); i++) {
            
            //if dealing with left child
            if (nodeIndex == 0) {
                
                //if value of splitting feature less than threshold, belongs in node
                if (data[i][seg.first] < seg.second) {
                    
                    //store data point and label info
                    tData.push_back(data[i]);
                    tLabels.push_back(labels[i]);
                }
            }
            
            //if dealing with right child
            else {
                
                //if value of splitting feature geq than threshold, belongs in node
                if (data[i][seg.first] >= seg.second) {
                    
                    //store data point and label info
                    tData.push_back(data[i]);
                    tLabels.push_back(labels[i]);
                }
            }
        }
    }
    
    //*** can optimize further by sorting points based on feature val, not using linked lists
    //returns feature index and threshold for best split for data in single node
    pair<int, int> DecisionTree::findSegmentor(Matrix& data, vector<int>& labels)
    {
        double entropy = 0.0;
        double minEntropy = 10.0;
        list<int>::iterator it;
        list<int> leftLabStore, rightLabStore;
        pair<int, int> p;
        
        
        //either check all features or a randomly selected subset
        std::vector<int> featureIndices;
        if (_nConsideredFeatures == _nFeatures) {
            
            //include all features (all indices)
            for (int i=0; i<_nFeatures; i++) {
                featureIndices.push_back(i);
            }
        } else {
            
            //randomly choose subset of feature indices
            srand (time(NULL));
            for (int i=0; i<_nConsideredFeatures; i++) {
                featureIndices.push_back(rand() % _nFeatures);
            }
        }
    
        
        //iterate through all features
        for (unsigned int i=0; i<featureIndices.size(); i++) {
            
            int m = featureIndices[i];
            
            
            //check for simple case of only two feature values (e.g. 0's and 1's), not split yet
            if (_featureValues[m].size() == 2) {
                
                leftLabStore.clear();
                rightLabStore.clear();
                
                //iterate through samples and check labels
                for (unsigned int k=0; k<data.size(); k++) {
                    
                    //just split into two based on value
                    if (data[k][m] == _featureValues[m].front()) {
                        leftLabStore.push_back(labels[k]);
                    } else {
                        rightLabStore.push_back(labels[k]);
                    }
                }
                //calculate entropy associated with split, store with feature and threshold
                entropy = calculateEntropy(leftLabStore, rightLabStore);
                
                //update best split if entropy is current minimum
                if (entropy < minEntropy) {
                    p.first = m;
                    p.second = _featureValues[m].back();
                    minEntropy = entropy;
                }
                
                leftLabStore.clear();
                rightLabStore.clear();
                
                
            //if more than two possible labels, have to iterate through possible splits
            }
            else if (_featureValues[m].size() > 2) {
            
                //iterate through all possible splits betwee observed values
                for (it = _featureValues[m].begin(); it != _featureValues[m].end(); ++it) {
                    
                    leftLabStore.clear();
                    rightLabStore.clear();
                
                    //iterate through samples and collect labels on either size of threshold
                    for (unsigned int k=0; k<data.size(); k++) {
                    
                        //check threshold: greater than / eq to j, less than j (binary splits)
                        if (data[k][m] < *it) {
                            leftLabStore.push_back(labels[k]);
                        } else {
                            rightLabStore.push_back(labels[k]);
                        }
                    }
                
                    //calculate entropy associated with that split
                    entropy = calculateEntropy(leftLabStore, rightLabStore);
                    
                    //update best split if entropy is current minimum
                    if (entropy < minEntropy) {
                        p.first = m;
                        p.second = *it;
                        minEntropy = entropy;
                    }
                    
                    leftLabStore.clear();
                    rightLabStore.clear();
                }
            }
        }
        //return <feature index, threshold> resulting in lowest entropy
        return p;
    }
    
    //calculates shannon entropy associated with a given split
    double DecisionTree::calculateEntropy(list<int>& l, list<int>& r)
    {
        double entropy = 0.0;
        double invL = 1.0/((double)l.size());
        double invR = 1.0/((double)r.size());
        double invTotalSize = 1.0/((double)(l.size() + r.size()));
        
        list<int>::iterator it;
        map<int, double> lHist, rHist;
        map<int, double>::iterator mapit;
        
        //make histogram of labels for each proposed node
        for (it = l.begin(); it != l.end(); ++it) {
            lHist[*it] += invL;
        }
        for (it = r.begin(); it != r.end(); ++it) {
            rHist[*it] += invR;
        }
        
        //calculate entropy for each proposed node
        double eL = 0.0;
        for (mapit = lHist.begin(); mapit != lHist.end(); ++mapit) {
            eL -= (mapit->second)*log2(mapit->second);
        }
        double eR = 0.0;
        for (mapit = rHist.begin(); mapit != rHist.end(); ++mapit) {
            eR -= (mapit->second)*log2(mapit->second);
        }
        
        //calculate total weighted entropy
        double fractL = ((double)l.size())*invTotalSize;
        double fractR = ((double)r.size())*invTotalSize;
        entropy = fractL*eL + fractR*eR;
        
        return entropy;
    }
    
    //randomizes samples+labels
    void DecisionTree::randomizeSamples(Matrix& data, std::vector<int>& labels)
    {
        Matrix data2;
        vector<int> labels2;
        srand(time(NULL));
        
        //choose random indices and add to new data structures
        for (unsigned int i=0; i<data.size(); i++) {
            
            int randNum = rand() % data.size();
            
            data2.push_back(data[randNum]);
            labels2.push_back(labels[randNum]);
            
            data.erase(data.begin() + randNum);
            labels.erase(labels.begin() + randNum);
        }
        
        //assign to original data structures
        data = data2;
        labels = labels2;
        data2.clear();
        labels2.clear();
    }
    
    //returns true if all samples in node have same label
    bool DecisionTree::sameLabels(vector<int>& trainLabels)
    {
        int val = trainLabels[0];
        for (unsigned int i=1; i<trainLabels.size(); i++) {
            if (val != trainLabels[i]) {
                return false;
            }
        }
        return true;
    }
    
    //returns index of feature passed as string
    int DecisionTree::getFeatureIndex(string sf)
    {
        for (unsigned int i=0; i<_nFeatures; i++) {
            if (_features[i] == sf) {
                return i;
            }
        }
        return -1;
    }
    
    //returns map of integer indices for each string feature
    void DecisionTree::makeFeatureIndexMap(vector<string>& sf)
    {
        for (unsigned int i=0; i<_nFeatures; i++) {
            _featureMap[_features[i]] = i;
            if (_vocal) cout << _features[i] << endl;
        }
    }
    
    //makes list of possible label values (assumes integers)
    list<int> DecisionTree::getLabelValues(vector<int>& labels)
    {
        list<int> l;
        
        //iterate through labels to find unique values
        for (unsigned int i=0; i<labels.size(); i++) {
            l.push_back(labels[i]);
            if (l.size() > 1) {
                l.sort();
                l.unique();
            }
        }
        return l;
    }
    
    //returns most probable label to use as default label
    int DecisionTree::getLabelMode(vector<int>& labels)
    {
        map<int, int> count;
        map<int, int>::iterator mapit;
        list<int>::iterator it;
        
        //iterate through each possible label
        for (it = _labelValues.begin(); it != _labelValues.end(); ++it) {
            
            //iterate through all training labels
            for (unsigned int i=0; i<labels.size(); i++) {
                
                if (labels[i] == *it) {
                    count[*it]++;
                }
            }
        }
        //iterate through counts for each label to get max
        int max = 0;
        int d = 0;
        for (mapit = count.begin(); mapit != count.end(); ++mapit) {
            if (mapit->second > max) {
                max = mapit->second;
                d = mapit->first;
            }
        }
        count.clear();
        return d;
    }
    
    //makes list of values for each feature
    vector< list<int> > DecisionTree::getFeatureValues(Matrix& data)
    {
        Matrix::iterator it;
        list<int> vals;
        vector< list<int> > info;
        
        //iterate through all features
        for (unsigned int i=0; i<_nFeatures; i++) {
            
            //iterate through all samples
            for (unsigned int j=0; j<data.size(); j++) {
                
                //store value of "i"th feature for "j"th sample
                vals.push_back(data[j][i]);
            }

            //store only unique features
            vals.sort();
            vals.unique();
            
            //add to info vector
            info.push_back(vals);
            vals.clear();
        }
        vals.clear();
        return info;
    }
}

