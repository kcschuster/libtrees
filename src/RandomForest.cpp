/*
 *  RandomForest.cpp
 *
 *  Random Forest: makes many decision trees using bootstrapping and
 *      randomized feature selection; classifies based on mode of 
 *      classification results
 *
 *  Assumptions/Methods for Random Forest (Decision Trees):
 *      (1) integer label values
 *      (2) binary splits
 *      (3) for given integer threshold value t: split [ , t) & [t, ]
 *      (4) assumes all feature values are integers
 *      (5) stops when data in each node has same label or reaches
 *          minimum allowed size (set by user)
 *
 *  Created by Kelsey Schuster
 *  11/5/15
 */


#include "RandomForest.h"

using namespace std;



namespace trees 
{
    RandomForest::RandomForest(vector<string>& features)
    : DecisionTree(features)
    {
        _storeHeadNodeSplits = false;
    }
    
    //sets whether we're storing splits made at head nodes
    void RandomForest::storeHeadNodeSplits()
    {
        _storeHeadNodeSplits = true;
    }
    
    //build random forest by bootstrapping and making multiple trees
    void RandomForest::trainRandomForest(Matrix& trainData, vector<int>& trainLabels, int nSamps, int nFeat, int minSize)
    {
        _nBootSamps = nSamps;
        _nConsideredFeatures = nFeat;
        _minNodeSize = minSize;
        
        _headNodeSplitStore.clear();
        
        
        //make bootstrap samples and train 1 tree for each sample
        for (unsigned int i=0; i<_nBootSamps; i++) {
            
            if (_vocal) cout << "decision tree " << i << endl;
            
            //get bootstrap sample
            Matrix sample;
            vector<int> sampLabels;
            getBootstrapSample(trainData, trainLabels, sample, sampLabels);
            
            //train decision tree
            DecisionTree::trainDecisionTree(sample, sampLabels, minSize);
            _treeStorage.push_back(_root);
            
            //if set, store head node split
            if (_storeHeadNodeSplits) {
                storeHeadNodeData();
            }
        }
    }
    
    //stores the split and threshold for the head node
    void RandomForest::storeHeadNodeData()
    {
        string feature = _features[_root->spltRule.first];
        int thresh = _root->spltRule.second;
        _headNodeSplitStore[feature][thresh]++;
    }
    
    //returns histogram of head node splits
    map<string, map<int, int> > RandomForest::getHeadNodeSplits()
    {
        return _headNodeSplitStore;
    }
    
    //returns bootstrap sample from original training set
    void RandomForest::getBootstrapSample(Matrix& trainData, vector<int>& trainLabels, Matrix& sample, vector<int>& sampLabels)
    {
        //randomly pick n random indices (n=trainData.size())
        int n = trainData.size();
        
        //initialize random number generator
        srand (time(NULL));
        
        //randomly choose n indices and put in sample matrix
        for (unsigned int i=0; i<n; i++) {
            int randNum = rand() % n;
            sample.push_back(trainData[randNum]);
            sampLabels.push_back(trainLabels[randNum]);
        }
    }
    
    //makes prediction with each model, takes mode of predictions
    void RandomForest::makePredictions(Matrix& testData, vector<int>& predictions)
    {
        vector< vector<int> > predictionStore;
        
        //iterate through each tree and make predictions
        for (unsigned int i=0; i<_treeStorage.size(); i++) {
            
            _root = _treeStorage[i];
            
            vector<int> p = vector<int>(testData.size(), 0);
            DecisionTree::makePredictions(testData, p);
            
            predictionStore.push_back(p);
            p.clear();
        }
        
        //iterate through each sample
        for (int j=0; j<predictionStore[0].size(); j++) {
            
            vector<int> results;
            
            //iterate through results from each tree
            for (unsigned int i=0; i<predictionStore.size(); i++) {
                
                results.push_back(predictionStore[i][j]);
            }
            
            //get mode of labels for prediction
            predictions[j] = DecisionTree::getLabelMode(results);
            results.clear();
        }
        predictionStore.clear();
    }
}
