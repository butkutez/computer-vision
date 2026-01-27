# computer-vision

# Pneumonia challenge

- Repository: `computer vision`
- Type of Challenge: `Consolidation`
- Duration: `2 days`
- Deadline: `dd/mm/yy H:i AM/PM`
- Deployment strategy :
  - Github page
- Team challenge : `duo`


## Mission objectives

- To learn how to design and evaluate a custom made convolutional neural network for practical purposes

## The Mission

________

<img src="https://media.giphy.com/media/Q3tOCfoaeh1qo/giphy.gif" align="right" width="250" />

Alright, you did it! You got the job! You've been hired as a data scientist at your dream company!

You read over your job description a final time while your partner is gushing over this wonderful opportunity:

> That's fantastic honey, we'll never have to worry about bills ever again!

It starts to dawn on you, your job description doesn't say 'data scientist' at all...it doesn't even seem data related...

"Radiologist"

Uh-oh. Time to fake it till you make it.

**Design a CNN capable of recognizing pneumonia in x-rays of patients**
_________


### Must-have features

- A CNN trained on [this dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) that can recognize new images outside of the training set
- Proper model evaluation (split dataset, confusion matrix, hyper-parameter tuning, etc)
- Visualizations of model results and evaluations (properly labeled, titled...)


### Nice-to-have features

- A visualization of the feature maps of the model
- Comparison with other CNN model structures


## Deliverables

1. Publish your source code on the GitHub repository.
2. Pimp up the README file:
   - Description
   - Installation
   - Usage
   - (Visuals)
   - (Contributors)
   - (Timeline)
   - (Personal situation)


### Steps

1. Start downloading the files
2. Analyze your folder contents; you'll notice a train, validation and test set
3. Do proper preprocessing steps using OpenCV (which steps make sense, which don't, discuss these with your colleagues)
4. Import and existing model from the [keras models](https://keras.io/api/applications/)
5. Adjust (or add) the output layer to only contain 2 classes, since we have two classes to identify: pneumonia or not
6. After building your model, it's time to train them. For a pre-existing model, you might want to first train your added layers by freezing the existing layers. Use the [transfer learning guide](https://keras.io/guides/transfer_learning/) to help you set up an efficient workflow. 
7. You will use the training set to train, after which you will use your validation set to see what the actual accuracy might be. If it is not sufficient, you will need to adjust your model hyper-parameters (learning rate, number of layers, nodes,...) and train again. Depending on your transfer learning set-up, your training time might be minimal, and you can test more hyper-parameter options (random search, grid search...)
8. Once your model accuracy is sufficient (you choose what you deem sufficient), you can test on your testing set. You do not use your test set to further tune your hyper-parameters
9. Plot all the relevant evaluation metrics for your model (confusion matrix, ROC, precision/recall, validation accuracy,...)
10. Discuss which evaluation metrics are the most important for this challenge (a hospital environment with lives on the line has different demands...)


## Evaluation criteria

| Criteria       | Indicator                                                                             | Yes/No |
|----------------|---------------------------------------------------------------------------------------|--------|
| 1. Is complete | You can show us the model results and your attempt at a CNN |        |
|                |                                            |
| 2. Is Correct  | Your model is properly evaluated and you can show your hyper-parameter tuning process (and explain your chosen parameters)  |        |
|                |  |        |
| 3. Is great         | You |        |
|                | |        |

## A final note of encouragement

Let's get that bread.

![You've got this!](https://media.giphy.com/media/VbzpvRC1LWGRQAvues/giphy.gif)



STREAMLIT LINK: https://computer-vision-pneuma-ai.streamlit.app/