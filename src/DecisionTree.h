/*
 *  DecisionTree.h
 *
 *  Decision Tree class: makes decision tree based on training data and returns
 *      predictions on test data.
 *
 *  Created by Kelsey Schuster
 *  11/3/15
 */

#ifndef DecisionTree_H
#define DecisionTree_H

#include <fstream>
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <list>

using namespace std;



namespace trees { 
    
    class DecisionTree
    {
        
    public:
        
        //constructor/destructor
        DecisionTree(vector<string>& features);
        ~DecisionTree();
        
        //data structure definitions
        typedef vector< vector<int> > Matrix;
        typedef map<int, vector< pair<string, vector<int> > > > SplitMap;
        typedef vector< pair<string, vector<int> > > SplitVector;
        typedef pair<string, vector<int> > SplitPair;
        
        
        //define a node class (for each node in the tree)
        struct Node
        {
            Node();
            ~Node();
            Node(int n=2);
            Node(const Node&);
            const Node& operator=(const Node&);
            
            pair<int, int> spltRule;   //pair of feature index (of _features) & threshold value to split at
            vector<Node*> chld;        //child nodes
            bool isLeaf;               //tells whether node is leaf node
            int lab;                   //when leaf node, label with which to classifty data points
        };
        
        //functions
        void trainDecisionTree(Matrix&, vector<int>&, int minSize=20);
        map<int, double> performCrossValidation(Matrix&, vector<int>&, vector<int>&, int param=1, int k=10);
        void makePredictions(Matrix&, vector<int>&);
        double computeValidationAccuracy(vector<int>&, vector<int>&);
        void setVocal(bool);
        void followSample(int);
        SplitVector getSampleSplits(int);
        
        
        
    protected:
        
        //global variables
        vector<string> _features;               //names of features
        int _nFeatures;                         //number of features
        int _nConsideredFeatures;               //# of features to use at each node
        int _minNodeSize;                       //min # samples allowed in a node
        vector< list<int> > _featureValues;     //observed values for each feature
        list<int> _labelValues;                 //possible labels based on training data
        int _defaultLabel;                      //most frequent label in training data
        map<string, int> _featureMap;           //for each string feature, gives index in feature vector
        Node* _root;                            //classification tree
        bool _vocal;                            //sets whether program prints out info
        int _followSampleIndex;                 //sample index whose splits we record
        SplitMap _sampleStore;                  //stores splits made for selected sample
        
        //functions
        void makeFeatureIndexMap(vector<string>&);
        list<int> getLabelValues(vector<int>&);
        int getLabelMode(vector<int>&);
        vector< list<int> > getFeatureValues(Matrix&);
        int getFeatureIndex(string);
        Node* buildDecisionTree(Matrix&, vector<int>&, Node*);
        bool sameLabels(vector<int>&);
        pair<int, int> findSegmentor(Matrix&, vector<int>&);
        double calculateEntropy(list<int>&, list<int>&);
        void trimMatrix(Matrix&, vector<int>&, Matrix&, vector<int>&, pair<int, int>&, int);
        void randomizeSamples(Matrix&, vector<int>&);
        void saveSplitInfo(int, int, int, int);
    };
}

#endif