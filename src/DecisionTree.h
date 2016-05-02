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



namespace trees { 
    
    class DecisionTree
    {
        
    public:
        
        //constructor/destructor
        DecisionTree(std::vector<std::string>& features);
        virtual ~DecisionTree();
        
        //data structure definitions
        typedef std::vector< std::vector<int> > Matrix;
        typedef std::pair<std::string, std::vector<int> > SplitPair;
        typedef std::vector< SplitPair > SplitVector;
        typedef std::map<int, SplitVector> SplitMap;
        
        
        //define a node class (for each node in the tree)
        struct Node
        {
            Node();
            ~Node();
            Node(int n=2);
            Node(const Node&);
            const Node& operator=(const Node&);
            
            std::pair<int, int> spltRule;   //pair of feature index (of _features) & threshold value to split at
            std::vector<Node*> chld;        //child nodes
            bool isLeaf;               //tells whether node is leaf node
            int lab;                   //when leaf node, label with which to classifty data points
        };
        
        //functions
        void trainDecisionTree(Matrix&, std::vector<int>&, int=20);
        std::map<int, double> performCrossValidation(Matrix&, std::vector<int>&, std::vector<int>&, int=1, int=10);
        virtual void makePredictions(Matrix&, std::vector<int>&);
        double computeValidationAccuracy(std::vector<int>&, std::vector<int>&);
        void setVocal(bool);
        void followSample(int);
        SplitVector getSampleSplits(int);
        
        
        
    protected:
        
        //global variables
        std::vector<std::string> _features;             //names of features
        int _nFeatures;                                 //number of features
        int _nConsideredFeatures;                       //# of features to use at each node
        int _minNodeSize;                               //min # samples allowed in a node
        std::vector< std::list<int> > _featureValues;   //observed values for each feature
        std::list<int> _labelValues;                    //possible labels based on training data
        int _defaultLabel;                              //most frequent label in training data
        std::map<std::string, int> _featureMap;         //for each string feature, gives index in feature vector
        Node* _root;                                    //classification tree
        bool _vocal;                                    //sets whether program prints out info
        int _followSampleIndex;                         //sample index whose splits we record
        SplitMap _sampleStore;                          //stores splits made for selected sample
        
        //functions
        void makeFeatureIndexMap(std::vector<std::string>&);
        std::list<int> getLabelValues(std::vector<int>&);
        int getLabelMode(std::vector<int>&);
        std::vector< std::list<int> > getFeatureValues(Matrix&);
        int getFeatureIndex(std::string);
        Node* buildDecisionTree(Matrix&, std::vector<int>&, Node*);
        bool sameLabels(std::vector<int>&);
        std::pair<int, int> findSegmentor(Matrix&, std::vector<int>&);
        double calculateEntropy(std::list<int>&, std::list<int>&);
        void trimMatrix(Matrix&, std::vector<int>&, Matrix&, std::vector<int>&, std::pair<int, int>&, int);
        void randomizeSamples(Matrix&, std::vector<int>&);
        void saveSplitInfo(int, int, int, int);
    };
}

#endif