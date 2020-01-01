"""
Last revised 7/30/2019

A class to generate and manage (including plotting) training data with a desired probability distribution, 
meant for neural networks that perform classification (object-type recognition.)

The main class, "trainingDataGenerator", provides support and common operations 
to assist helper classes that are specific to desired SIMULATED ENVIRONMENTS.

5 sample helper classes are included.

Helper classes (named "environmentSimulator_n" for some n) can use arbitrary code
to implement a simple interface - basically a mapping
from feature values to a probability of a given Object type.

* The class "trainingDataGenerator" provides a common API for all environments,
  and provides convenient common features such as data generation,
  association of feature values with true Object type,
  distribution previews and validation,
  plotting and histograms.

The 5 included sample helper classes generate training data sets for 5 SIMULATED ENVIRONMENTS:

      * Environments 1 and 2 are deterministic, where the value of the single feature, ranging in the [0, 1] interval,
      uniquely determines the object type.

      * Environments 3, 4 and 5 are probabilistic: each value of the feature corresponds to a set of probabilities
      for the various object types.


USAGE EXAMPLE

    trainData = trainingDataGenerator(simID=3, nTrainingPoints=5000, randomSeed=13) # This will use the helper class "environmentSimulator_3"
    
    # Print and plot a little summary of the object-type probabilities as x varies in the [0,1] range
    trainData.valuesAcrossRange()
    
    # Show a histogram of the data
    trainData.plotDatasetHistogram(200)


HELPER CLASSES for specific Simulated Environments

    Each helper class needs to provide 2 methods: 
	
		1 - nClasses()  : it returns the number of distinct Object types simulated by this environment
		
		2 - objTypeProbabilities(objType, x) : Given a value x, or set of values, for a feature, 
		                     it returns a probability, or set of probabilities, for the specified object type


DEPENDENCIES
    * numpy as np
    * matplotlib
    * sys


	----------------------------------------------------------------------------------
	MIT License
	Copyright (c) 2019 Julian A. West
	
	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:
	
	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.
	
	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
	WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF 
	OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
	----------------------------------------------------------------------------------	

"""


import numpy as np
import matplotlib.pyplot as plt
import sys



class trainingDataGenerator:
    """
    Class to generate and manage training data, consisting of a variety of simulated environments, with a single numeric features that specifies,
    either deterministically or probabilistically, one several classes (object types to distinguish)

    SIDE EFFECT: the Numpy random-number generator gets used, and it will affect any random number generated downstream
    Also, the NumPy random-number generator will get seeded (IF a randomSeed is passed)

    OBJECT ATTRIBUTES:
        all_xs:         nx1 Numpy matrix (column vector) of real numbers from the interval [0,1]
        all_ys:         Numpy array with n integers, each one between 0 and (nClasses-1), representing the "object types" (classes) of the corresponding points in all_xs
        trainingSize:   Number of total training points
        nClasses:       Integer specifying the number of classes (object types to distinguish).  Notice that different Environments may have different numbers of classes
    """

    def __init__(self, simID, nTrainingPoints, randomSeed=None):
        """
        Initialize the class and generate the training data.
        The number of classes (object types) is determined by the simulation ID

        :param simID:           An integer ID specifying which environment to use
        :param nTrainingPoints: Number of total training points (integer)
        :param randomSeed:      Optional seed to use for the random number generator
        """
        self.environmentID = simID      # This indicates which environment to use ("how to map a feature value into a true output object type")

        if randomSeed is not None:
            np.random.seed(randomSeed)  # Seed the NumPy random-number generator

        self.probabilitiesObj = None

        # Locate the Environment-specific helper class to use
        className:str = "environmentSimulator_" + str(simID)     # For example, the class name "environmentSimulator_1"

        try:
            envSpecificClass = self.___strToClass(className)    # For example, the class named "environmentSimulator_1"
        except:
            print("A class named '%s' should be present to handle the requested Environment ID %d, but none was found" % (className, simID))
            exit(1)

        # Instantiate the Environment-specific helper class
        self.probabilitiesObj = envSpecificClass()

        self.nClasses = self.probabilitiesObj.nClasses()    # Integer specifying the number of classes (object types to distinguish)
        self.trainingSize = nTrainingPoints                 # Number of total training points


        (self.all_xs, self.all_ys) = self.__generateTrainingData(nTrainingPoints, self.nClasses)
        #print(self.all_xs, self.all_ys)



    def simulateTrueObjectType(self, x):
        """
        Using the simulated environment specified in the object property "probabilitiesObj",
        map a Numpy array of floats from the [0,1] interval, representing a set of feature values,
        to a Numpy array of object types selected in accordance to the probability density functions of the object types.

        :param x:   a Numpy array of floats from the [0., 1.] range, representing a set of feature values
        :return:    a Numpy array of integers in the [0, self.nClasses-1] range
        """

        aux = self.probabilitiesObj     # An "auxiliary" Environment-specific object

        objTypeProbColumns = []         # A list of probability vectors for the various Object types
                                        #   (each vector, best visualized as a column, has as many elements as the number of feature values)

        # Loop over all classification types (which are in the range [0, self.nClasses-1]
        objectTypeNumber = 0
        while objectTypeNumber < self.nClasses:
            probsOfGivenObjectType = aux.objTypeProbabilities(objectTypeNumber, x)  # A Numpy array of floats from the [0., 1.] range, with same size as the array of feature values x
                                                                                    # For example, if x has size 4:  array([0.1 , 0.5, 0. ,  0.9])
                                                                                    # The entries are the probabilities of an Object of the current type (objectTypeNumber) at the various feature values
            objTypeProbColumns.append(probsOfGivenObjectType)       # Build up a List of those Numpy arrays: one for each Object Type
            objectTypeNumber += 1

        #print("objTypeProbColumns: ", objTypeProbColumns)

        probMatrix = np.column_stack(objTypeProbColumns)    # Treat the individual Numpy arrays in the List "objTypeProbColumns" as columns, and form a matrix from all of them.
                                                    # Every row of this matrix will give the probabilities of each Object type for a particular feature value (and ought to add up to 1.)
                                                    # The dimensions of this matrix are (number of feature values) x (number of Object types)
        #print("probMatrix:", probMatrix)

        allTrueObjectTypes = []         # List of integers in the [0, self.nClasses-1] range

        for row in probMatrix:    # Loop over the feature values
            #print("row:", row)      # Example for 6 Object types: [0.5  0.05 0.45 0.   0.   0.  ]    Note that the values ought to add up to 1.
            pickedObjType = np.random.choice(self.nClasses, p=row)      # THE HEAVY LIFTING OCCURS HERE: an object type is picked randomly based on the probability distribution of the various types
            #print("pickedObjType:", pickedObjType)
            allTrueObjectTypes.append(pickedObjType)

        #print("allTrueObjectTypes", allTrueObjectTypes)

        return np.array(allTrueObjectTypes)



    def getEntireTrainingSet(self):
        """
        Retrieve the full training dataset

        :return: A 2-tuple (all_x, all_y), where
                        all_x is an nx1 Numpy matrix (i.e. a "column vector")
                        all_y ia a Numpy array with n integers, each between 0 and (numberOfClasses-1)
        """
        return (self.all_xs, self.all_ys)



    def getBatch(self, batchSize):
        """
        Return a randomly-picked part of the (pre-computed) full training data

        :param batchSize:   An integer, specifying how many data points to retrieve
        :return:            A 2-tuple (xs, xs), where
                                xs is an nx1 Numpy matrix (i.e. a "column vector")
                                ys ia a Numpy array with n integers, each between 0 and (self.nClasses-1),
                                representing the "object types" (classes) of the corresponding points in all_xs
        """

        randomIndices = np.random.choice(self.trainingSize, batchSize, replace=False)

        return (self.all_xs[randomIndices],
                self.all_ys[randomIndices])



    def oneHotEncode(self, x, n_classes):
        """
        One hot encode a list of sample labels. Return a one-hot encoded vector for each label.

        :param x:           List of sample Labels
        :param n_classes:   An integer >= 1 indicating the number of distinct classes
        :return:            Numpy array of one-hot encoded labels
        Source: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python

        EXAMPLE:    oneHotEncode([0, 1, 2], 4)  will return:
                    array([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.]])
        """
        return np.eye(n_classes)[x]



    def plotDatasetHistogram(self, nbins = 30):
        """
        Plot a histogram of the dataset

        :param nbins:   Number of bins to use in the histogram
        :return: None
        """

        fig, ax = plt.subplots(self.nClasses, 1, tight_layout=True)


        for i in range(self.nClasses):
            # Extract all elements of all_xs for whom the corresponding element of all_ys has value equal to i
            xSubpop = self.all_xs[self.all_ys == i]         # Sub-population of x values
            # See: https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-92.php

            #print("x's for obj. type", i, ": ", xSubpop)
            ax[i].hist(xSubpop, bins=nbins, range=(0., 1.))
            ax[i].set(xlabel="Object type " + str(i))

        plt.show()



    def valuesAcrossRange(self):
        """
        Print a little summary of the object-type probabilities as x varies in the [0,1] range,
        as well as all the combined probability density PLOTS (in a pop-up window)

        :return: None
        """

        """
        A small sampling of the range of feature values for the printout
        """
        x = np.arange(0.0, 1.001, 0.1)

        print("-- Summary of object-type probabilities as feature x varies in the [0,1] range --")
        print("    x:        ", x, "\n")


        objectTypeNumber = 0
        total:float = 0.

        while objectTypeNumber < self.nClasses:
            probArray = self.probabilitiesObj.objTypeProbabilities(objectTypeNumber, x)  # Probabilities of this object type across the sampled feature values
            print("    Obj ", objectTypeNumber, " : ", probArray)   # Example:  "    Obj  0  :  [0.5 0.5 0.  0.  0.  0.  0.  0.  0.5 0.5 0.5]"
            total += probArray
            objectTypeNumber += 1

        print("    SUM:      ", total)     # This ought to be all 1.'s


        """
        A larger sampling of the range of feature values for the plot
        """
        x = np.arange(0.0, 1.001, 0.01)

        # Prepare plot labels/title
        plt.xlabel('x')
        plt.ylabel('Probabilities of each object type')
        plt.title("Probabilistic generation of the various object types\nas a function of a single feature - ENVIR. " + str(self.environmentID))


        objectTypeNumber = 0

        while objectTypeNumber < self.nClasses:
            probArray = self.probabilitiesObj.objTypeProbabilities(objectTypeNumber, x)  # Probabilities of this object type across the sampled feature values
            plt.plot(x, probArray)
            objectTypeNumber += 1


        # Prepare the plot legend; for example, for 6 object types, it'd be "012345".  NOTE: this only works for up to 10 classes
        caption = ""
        for i in range(self.nClasses):
            caption += str(i)


        plt.legend(caption)  # The legend command must appear after the plots were generated
        plt.show()




    #######################################   PRIVATE methods   ###############################################

    def __generateTrainingData(self, nTrainingPoints, numberOfClasses):
        """
        Create hypothetical training data: the specified number of training data points,
        mapped to the requested number of output classes (representing "object types" to distinguish)

        :param nTrainingPoints: Desired number of training data points
        :param numberOfClasses: Desired number of output classes (representing "object types" to distinguish)
        :return: A 2-tuple (all_x, all_y), where
                        all_x is an nx1 Numpy matrix (i.e. a "column vector")
                        all_y ia a Numpy array with n integers, each between 0 and (numberOfClasses-1)
        """

        # Create  list with the specified number of points, all with uniform distribution in [0,1]
        training_x = np.random.rand(nTrainingPoints)
        training_y = self.simulateTrueObjectType(training_x)  # training_y ia a Numpy array of floats

        all_x = training_x.reshape(-1, 1)  # The reshape creates an (nx1) matrix ("column vector")

        all_y = training_y

        return (all_x, all_y)



    def ___strToClass(self, className: str):
        """
        Return the class in the current module with the requested method name, if it exists.
        It will only work for the class defined in the current module

        Alternate implementation:  globals()[className]

        :param className:   A string with the name of the desired class
        :return:            The class in the current module with the requested method name, if it exists.
                            If it doesn't exist, throw an exeption
        """

        module = sys.modules[__name__]  # This will be an object such as <module '__main__' from 'somePath/thisFileName.py'>
        try:
            identifier = getattr(module, className) # This will be an object such as <class '__main__.environmentSimulator_5'>
        except AttributeError:
            raise NameError("No attribute named '%s' exists in this module." % className)

        # Make sure that the located attribute is indeed a class name
        if isinstance(identifier, type):
            return identifier

        raise TypeError("'%s' is not a class." % className)

######################## END OF CLASS "trainingDataGenerator" ########################################



######################################################################################################
#
#		BELOW ARE 5 SAMPLE HELPER CLASSES for specific Simulated Environments
#
#				TO BE DROPPED AND/OR MODIFIED AS NEEDED
#
#             Each helper class needs to provide 2 methods: nClasses() and objTypeProbabilities()
#
######################################################################################################


class environmentSimulator_1:
    """
    Environment 1 splits the [0-1] range of feature values into side-by-side segments that deterministically specify the Object type
    (i.e., for any feature value, one of the Object types will have probability 1, while all the other Object types have probability 0.)
    The probability density functions for the various Object types consists of adjacent rectangles all of height 1.
    """

    def __init__(self):
        self.n_classes = 6


    def nClasses(self):
        return self.n_classes


    def objTypeProbabilities(self, objType:int, x):
        """
        Given a value, or set of values, for a feature, generate a probability, or set of probabilities, for the specified object type

        :param objType: An integer between 0 and 5 (self.n_classes - 1)
        :param x:       Either a Numpy float, or a Numpy array of floats from the [0., 1.] range, representing multiple samplings of a single feature
        :return:        A Numpy array of floats from the [0., 1.] range, with the same number of elements as x
        """

        numberOfSegments = self.n_classes       # We are splitting the [0-1] feature range into a set of equal segments (intervals),
                                                # each of size 1./numberOfSegments

        # Loop over all possible Object type values (i.e. from 0 to self.n_classes-1)
        i = 0
        while i < self.n_classes:
            if objType == i:
                # The probability density of this object type consists of a rectangle of height 1, when x is in the i-th segment (counting from 0)
                # If x is below or above that segment, the probability is 0.; if it's within the segment, it's 1.
                return np.where(x < float(i) / numberOfSegments,
                            0.,
                            np.where(x >= float(i+1) / numberOfSegments, 0., 1.))

            i += 1

######################## END OF CLASS "environmentSimulator_1" ########################################



class environmentSimulator_2:
    """
    This helper class is an expansion of environmentSimulator_1, where the probability-density rectangle for object type 0 makes a "comeback" at the end of the [0-1] interval
    """

    def __init__(self):
        self.n_classes = 6


    def nClasses(self):
        return self.n_classes


    def objTypeProbabilities(self, objType:int, x):
        """
        Given a value, or set of values, for a feature, generate a probability, or set of probabilities, for the specified object type

        :param objType: An integer between 0 and 5 (self.n_classes - 1)
        :param x:       Either a Numpy float, or a Numpy array of floats from the [0., 1.] range, representing multiple samplings of a single feature
        :return:        A Numpy array of floats from the [0., 1.] range, with the same number of elements as x
        """

        numberOfSegments = self.n_classes + 1   # The +1 is to take care of the "comeback" at the end for Object Type 0

        if objType == 0:
            # The probability density of this object type consists of 2 rectangles of height 1, when x is in the first or sixth (last) segment
            return np.where(x < 1. / numberOfSegments,
                            1,
                            np.where(x >= float(self.n_classes) / numberOfSegments, 1.0, 0.))

        # Loop over all other possible Object type values (i.e. from 1 to self.n_classes-1)
        i = 1
        while i < self.n_classes:
            if objType == i:
                return np.where(x < float(i) / numberOfSegments,
                            0.,
                            np.where(x >= float(i+1) / numberOfSegments, 0., 1.))

            i += 1

######################## END OF CLASS "environmentSimulator_2" ########################################



class environmentSimulator_3:
    """
    This helper class generates a 6-object type data similar to ENVIRONMENT 2, but with 2 triangles splashed across the density functions
    """

    def __init__(self):
        self.n_classes = 6


    def nClasses(self):
        return self.n_classes


    def objTypeProbabilities(self, objType:int, x):
        """
        Given a value, or set of values, for a feature, generate a probability, or set of probabilities, for the specified object type

        :param objType: An integer between 0 and 5 (self.n_classes - 1)
        :param x:       Either a Numpy float, or a Numpy array of floats from the [0., 1.] range, representing multiple samplings of a single feature
        :return:        A Numpy array of floats from the [0., 1.] range, with the same number of elements as x
        """

        numberOfSegments = 5

        if objType == 0:
            # The probability density of this object type consists of 2 rectangles of height 0.5, one for x from 0 to 0.2, and one for x f from 0.8 to 1.0
            return np.where(x < 1. / numberOfSegments,
                            0.5,
                            np.where(x >= 4. / numberOfSegments, 0.5, 0.))

        if objType == 1:
            # The distribution of this object type consists of an upward-pointing triangle whose apex is at 0.5, and height l/2
            return np.where(x < 0.5,
                            x,
                            (1 - x))

        if objType == 2:
            # The distribution of this object type looks like a downward-pointing triangle, with the bottom at 0.5, and height l/2
            return np.array(0.5 - self.objTypeProbabilities(1, x))         # Note: without the np.array wrap, it seems to return a non-array if x has dim 1

        if objType == 3:
            # The probability density of this object type consists of a rectangle of height 0.5, for x from 0.2 to 0.4
            return np.where(x < 1. / numberOfSegments,
                            0.,
                            np.where(x >= 2. / numberOfSegments, 0., 0.5))

        if objType == 4:
            # The probability density of this object type consists of a rectangle of height 0.5, for x from 0.4 to 0.6
            return np.where(x < 2. / numberOfSegments,
                            0.,
                            np.where(x >= 3. / numberOfSegments, 0., 0.5))

        if objType == 5:
            # The probability density of this object type consists of a rectangle of height 0.5, for x from 0.6 to 0.8
            return np.where(x < 3. / numberOfSegments,
                            0.,
                            np.where(x >= 4. / numberOfSegments, 0., 0.5))

######################## END OF CLASS "environmentSimulator_3" ########################################



class environmentSimulator_4:
    """
    This helper class is a simplified version of ENVIRONMENT 3, for ONLY 2 OBJECT TYPES, and only retaining the 2 triangles splashed across the density functions
    """

    def __init__(self):
        self.n_classes = 2


    def nClasses(self):
        return self.n_classes


    def objTypeProbabilities(self, objType:int, x):
        """
        Given a value, or set of values, for a feature, generate a probability, or set of probabilities, for the specified object type

        :param objType: An integer between 0 and 1 (self.n_classes - 1)
        :param x:       Either a Numpy float, or a Numpy array of floats from the [0., 1.] range, representing multiple samplings of a single feature
        :return:        A Numpy array of floats from the [0., 1.] range, with the same number of elements as x
        """

        if objType == 0:
            # The distribution of this object type consists of an upward-pointing triangle whose apex is at 0.5, and height l
            return np.where(x < 0.5,
                            2*x,
                            (2 - 2*x))

        if objType == 1:
            # The distribution of this object type looks like a downward-pointing triangle, with the bottom at 0.5, and height l
            return np.array(1 - self.objTypeProbabilities(0, x))         # Note: without the np.array wrap, it seems to return a non-array if x has dim 1

######################## END OF CLASS "environmentSimulator_4" ########################################



class environmentSimulator_5:
    """
    This helper class generates a 6-object type data similar to ENVIRONMENT 3, but with 2 rectangles in lieu of the triangles (both splashed across the density functions: one at p=0.2 and one at p=0.3)
    """

    def __init__(self):
        self.n_classes = 6


    def nClasses(self):
        return self.n_classes


    def objTypeProbabilities(self, objType:int, x):
        """
        Given a value, or set of values, for a feature, generate a probability, or set of probabilities, for the specified object type

        :param objType: An integer between 0 and 5 (self.n_classes - 1)
        :param x:       Either a Numpy float, or a Numpy array of floats from the [0., 1.] range, representing multiple samplings of a single feature
        :return:        A Numpy array of floats from the [0., 1.] range, with the same number of elements as x
        """

        numberOfSegments = 5

        if objType == 0:
            # The probability density of this object type consists of 2 rectangles of height 0.5, one for x from 0 to 0.2, and one for x f from 0.8 to 1.0
            return np.where(x < 1. / numberOfSegments,
                            0.5,
                            np.where(x >= 4. / numberOfSegments, 0.5, 0.))

        if objType == 1:
            # The probability density of this object type consists of a rectangle of height 0.2, across the entire x span from 0 to 1
            return np.full(x.size, 0.2)      # A constant Numpy array with the same size as the input feature

        if objType == 2:
            # The probability density of this object type consists of a rectangle of height 0.3, across the entire x span from 0 to 1
            return np.full(x.size, 0.3)      # A constant Numpy array with the same size as the input feature

        if objType == 3:
            # The probability density of this object type consists of a rectangle of height 0.5, for x from 0.2 to 0.4
            return np.where(x < 1. / numberOfSegments,
                            0.,
                            np.where(x >= 2. / numberOfSegments, 0., 0.5))

        if objType == 4:
            # The probability density of this object type consists of a rectangle of height 0.5, for x from 0.4 to 0.6
            return np.where(x < 2. / numberOfSegments,
                            0.,
                            np.where(x >= 3. / numberOfSegments, 0., 0.5))

        if objType == 5:
            # The probability density of this object type consists of a rectangle of height 0.5, for x from 0.6 to 0.8
            return np.where(x < 3. / numberOfSegments,
                            0.,
                            np.where(x >= 4. / numberOfSegments, 0., 0.5))

######################## END OF CLASS "environmentSimulator_5" ########################################
