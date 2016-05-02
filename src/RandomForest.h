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
        RandomForest(std::vector<std::string>&);
        void trainRandomForest(Matrix&, std::vector<int>&, int=100, int=10, int=20);
        void makePredictions(Matrix&, std::vector<int>&);
        void storeHeadNodeSplits();
        std::map<std::string, std::map<int, int> > getHeadNodeSplits();
        
        
    protected:
        
        //global variables
        std::vector<Node*> _treeStorage;                                //store decision trees
        int _nBootSamps;                                                //number bootstrap samples (trees) to make
        bool _storeHeadNodeSplits;                                      //sets whether we save head node data
        std::map<std::string, std::map<int, int> > _headNodeSplitStore; //stores all head node splits
        
        //functions
        void getBootstrapSample(Matrix&, std::vector<int>&, Matrix&, std::vector<int>&);
        void storeHeadNodeData();
        
    };
}

#endif