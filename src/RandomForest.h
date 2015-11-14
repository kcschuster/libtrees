/*
 *  RandomForest.h
 *
 *  Random Forest: makes many decision trees using bootstrapping and
 *      randomized feature selection; classifies based on mode of 
 *      classification results
 *
 *  Created by Kelsey Schuster
 *  11/5/15
 */

#ifndef RandomForest_H
#define RandomForest_H

#include "DecisionTree.h"


namespace trees {
    
    class RandomForest : public DecisionTree
    {
    public:
        
        //functions
        RandomForest(vector<string>&);
        void trainRandomForest(Matrix&, vector<int>&, int nSamps=100, int nFeat=10, int minSize=20);
        void makePredictions(Matrix&, vector<int>&);
        void storeHeadNodeSplits();
        map<string, map<int, int> > getHeadNodeSplits();
        
        
    protected:
        
        //global variables
        vector<Node*> _treeStorage;                         //store decision trees
        int _nBootSamps;                                    //number bootstrap samples (trees) to make
        bool _storeHeadNodeSplits;                          //sets whether we save head node data
        map<string, map<int, int> > _headNodeSplitStore;    //stores all head node splits
        
        //functions
        void getBootstrapSample(Matrix&, vector<int>&, Matrix&, vector<int>&);
        void storeHeadNodeData();
        
    };
}

#endif