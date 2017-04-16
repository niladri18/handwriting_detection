# Table of Contents

1. [Handwriting Detection](README.md#problem-summary)
2. [Details of implementation:](README.md#details-of-implementation)

# Problem summary

A set of training examples that contains 5000 handwritten digits is taken. We will run a linear logistic regression model which will detect the handwriting predict numbers after reading.



# Details of implementation

Each example is is 20 pixel by 20 pixel image of the digit. The 20x20 grid of pixels (represented by floating point numbers) is unrolled into a 400-dimensional vector. Each example will become a single row with 400 entries each. Since number of training examples is 5000, this will lead to formation of a 5000x400 matrix. The last column of each row is the label (classification) of the training set. 

## How to run the code:

#### Step 1
Clone this git repository. 

#### Step 2
After the download is complete, you will find a folder named `/handwriting detection` in your local drive. Enter this repository.

#### Step 3
Place the input file in `/input` folder. A sample input file is given `/input/input.txt`

#### Step 4
While staying in the  `/handwriting detection` folder, type the following command:

For Linux:

`~$bash run.sh` 
 
 For Windows:
 
 `run.sh`
 
 After completion, the trained parameters will be saved in `/output` folder.

