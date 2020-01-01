# Training Data Generator and Manager for Classification Machine Learning

A class to generate and manage (including plotting) **training data with a desired probability distribution**, 
meant for neural networks that perform *classification (object-type recognition.)*


The main class, "**trainingDataGenerator**", provides support and common operations 
to assist helper classes that are specific to desired SIMULATED ENVIRONMENTS.  

*5 sample helper classes are included.*

Helper classes (named "**environmentSimulator_n**" for some *n*) can use arbitrary code
to implement a simple interface - basically a mapping
from feature values to probabilities of a given Object type.


* The class "**trainingDataGenerator**" provides a common API for all environments,
  and provides convenient common features such as data generation,
  association of feature values with true Object type,
  distribution previews and validation,
  plotting and histograms.

The 5 included sample helper classes generate training data sets for 5 *SIMULATED ENVIRONMENTS*:

      * Environments 1 and 2 are deterministic, where the value of the single feature, ranging in 
      the [0, 1] interval, uniquely determines the object type.

      * Environments 3, 4 and 5 are probabilistic: each value of the feature corresponds to 
      a set of probabilities for the various object types.


BASIC USAGE EXAMPLE

    trainData = trainingDataGenerator(simID=3, nTrainingPoints=10000, randomSeed=13) # This will use the helper class "environmentSimulator_3"
    
    # Print and plot a little summary of the object-type probabilities as x varies in the [0,1] range
    trainData.valuesAcrossRange()   # Notice how, for any x value, all the probabilities always add up to 1.
   [*Plot*](https://github.com/JuliansPolymathExplorations/Training-Data-Generator-and-Manager-for-Classification-ML/blob/master/screenshots/Env%203.png)
    
    # Show a histogram of the data
    trainData.plotDatasetHistogram(200)  
   [*Histogram (using a larger number of Training Points)*](https://github.com/JuliansPolymathExplorations/Training-Data-Generator-and-Manager-for-Classification-ML/blob/master/screenshots/Env%203%20histogram.png)


HELPER CLASSES for specific Simulated Environments

	Each helper class needs to provide 2 methods:

	    1 - nClasses()  : it returns the number of distinct Object types simulated by this environment

	    2 - objTypeProbabilities(objType, x) : Given a value x, or set of values, for a feature,
		                   it returns a probability, or set of probabilities, for the specified object type


DEPENDENCIES

    * numpy
    
    * matplotlib
    
    * sys
    
 
 ADDITIONAL USAGE EXAMPLE

    trainData = trainingDataGenerator(simID=1, nTrainingPoints=10000, randomSeed=13) 
    # This will use the helper class "environmentSimulator_1", a simple deterministic distribution
    # where the [0-1] interval is split into 6 equal segments, assigned to each of the 6 Object types
   [*Plot*](https://github.com/JuliansPolymathExplorations/Training-Data-Generator-and-Manager-for-Classification-ML/blob/master/screenshots/Env%201.png)
    
    print("Total number of Object types:", trainData.nClasses)          # It'll show 6
    
    # Print and plot a little summary of the object-type probabilities as x varies in the [0,1] range
    trainData.valuesAcrossRange()  
    
    """ It will show the following (notice how the object type gradually changes from 0 to 5,
    as x ranges from 0. to 1.:
    
    -- Summary of object-type probabilities as feature x varies in the [0,1] range --
    x:         [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ] 

    Obj  0  :  [1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    Obj  1  :  [0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
    Obj  2  :  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
    Obj  3  :  [0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0.]
    Obj  4  :  [0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]
    Obj  5  :  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
    SUM:       [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]
    """

    
    # Show a histogram of the data
    trainData.plotDatasetHistogram(200)  
   [*Histogram (from a run with more data)*](https://github.com/JuliansPolymathExplorations/Training-Data-Generator-and-Manager-for-Classification-ML/blob/master/screenshots/Env%201%20histogram.png) 
    
    batch = trainData.getBatch(5)   # This is a 2-tuple comprising a Numpy array of feature values, 
                                    # and a Numpy array of the corresponding true Object types
    print(batch)
    
    """ This shows:
    (array([[0.17],
       [0.01],
       [0.1 ],
       [0.63],
       [0.1 ]]), 
       
      array([1, 0, 0, 3, 0])
     )
    """
    
    fullDataSet = trainData.getEntireTrainingSet()
    print(fullDataSet)
    
    """ This shows (the very-abridged) output:
    (array([[0.78],
       [0.24],
       [0.82],
       ...,
       [0.13],
       [0.8 ],
       [0.3 ]]), 
       
      array([4, 1, 4, ..., 0, 4, 1])
     )
    """
    
    
    print(trainData.oneHotEncode([0, 1, 2], 4))
    
    """ This shows the NumPy matrix:
    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]]
    """

    trueObjectTypes = trainData.simulateTrueObjectType(np.array([.1, .3, .4, .5, .75, .9]))
    print(trueObjectTypes)   # This will display:  [0 1 2 3 4 5]    NOTE: for this enviroment, the values are deterministic
    
    # DETERMINISTIC vs. PROBABILISTIC
    
    # If we switch from the deterministic Simulated Environment 1, used above, 
    # to a non-deterministic one, such as Simulated Environment 5,
    # then the simulated True Object Type may vary even if the feature value is kept constant:
    
    trainData = trainingDataGenerator(simID=5, nTrainingPoints=10000, randomSeed=13)  # Switching to a PROBABILISTIC environment
    trueObjectTypes = trainData.simulateTrueObjectType(np.full(50, 0.5))    # 50 identical 0.5 values
    print(trueObjectTypes)        
    # It shows: [2 2 4 2 2 4 2 4 2 4 2 4 4 4 4 2 4 4 4 4 2 4 4 4 1 2 4 2 2 2 1 4 2 2 4 1 4 2 4 2 2 4 4 2 4 2 4 2 4 1]
    
    # If you look at the density-distribution function for this Enviroment 5, you will notice
    # how, when x = 0.5, Object types (1, 2, 4) are possible, with 4 being the most common
    # and 1 the least common:
    trainData.valuesAcrossRange() 
   [*plot*](https://github.com/JuliansPolymathExplorations/Training-Data-Generator-and-Manager-for-Classification-ML/blob/master/screenshots/Env%205.png)
