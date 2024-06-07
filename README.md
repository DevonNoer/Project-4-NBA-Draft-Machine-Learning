# Project-4-NBA-Draft-Machine-Learning
## Intro to the project
Our goal for this project was to find a dataset and create a model using machine learning to get an accuracy score of over 80. We chose a dataset that had college basketball stats and info and whether or not they were drafted. We decided to try and create the best model we could to predict if someone was going to get drafted or not. We started by loading in our csv and dropping columns that had too many null values, and cleaning up the data in general. Once we cleaned our data the next step was to get it into spark, so we could pull out what we wanted in our models using querys. After we had our query's all that was left was to build and optimize our models.
## Model 1
The first model was a neural network with the features(conf, GP, mp, pts, Ortg, usg, TS_per, FTM, FTA, twoPA, twoPM, TPM, TPA, drtg, bpm, oreb, dreb, ast, stl, blk, drafted). This model had two hidden layers, one with the activation function tanh, and the other with relu. The output layer had the activation function of sigmoid. The hidden layers had 12 and 6 neurons respectivly. This was the classification report for this model.               precision    recall  f1-score    <br>
<br>
                                                                                       Undrafted       0.98      0.96      0.97     <br>
                                                                                         Drafted       0.17      0.34      0.23     <br> 
                                                                                        accuracy                           0.94     <br>
