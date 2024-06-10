# Project-4-NBA-Draft-Machine-Learning

### Project by:
* Devon Noer
* Jack Maxwell
* Gabby Kruger
* Justyn Helgeson


## Introduction to the Project and Dataset
Our goal for this project was to find a dataset and create a model using machine learning to get an accuracy score of over 80. We chose a dataset that had college basketball stats and info and whether or not they were drafted. We decided to try and create the best model we could to predict if someone was going to get drafted or not. We started by loading in our csv and dropping columns that had too many null values, and cleaning up the data in general. Once we cleaned our data the next step was to get it into spark, so we could pull out what we wanted in our models using querys. After we had our query's all that was left was to build and optimize our models.

## Overview of files
Code :<br>
- Project_4.ipynb : Contains the code for cleaning and transforming our data, also our first six models. <br>
- 2024Draft_Project.ipynb : Contains the code with model six, and testing on a new datasets with 2023-2024 stats.<br>
- Web_Scrape.ipynb : Contains the code for the web scraping used on ESPN.com.<br>
Images : This folder contains images of our graphs relating to how well each model performed.<br>

Resources :<br>
- CollegeBasketballPlayers2009-2021.csv : This is the data that we used to build all of our models off of.<br>
- fixed_new_draft_2024.csv : This is the data that we used to predict who would get drafted in model six.<br>
- top750.csv : Player Data from ESPN college players.<br>
- Cleaned_df.csv : Cleaned Data used after cleaning CollegeBasketballPlayers2009-2021.csv.<br>
- cleaned_og.csv : Cleaning Data from Cleaned_df used for second draft attempt.<br>
- modified_data.csv : Player Data used from reddit excel file.<br>

## Model 1
The first model was a neural network with the features(conf, GP, mp, pts, Ortg, usg, TS_per, FTM, FTA, twoPA, twoPM, TPM, TPA, drtg, bpm, oreb, dreb, ast, stl, blk, drafted). This model had two hidden layers, one with the activation function tanh, and the other with relu. The output layer had the activation function of sigmoid. The hidden layers had 12 and 6 neurons respectivly. This was the classification report for this model, Undrafted{precision : 0.98, recall : 0.96, f1-score : 0.97}, Drafted{precision : 0.17, recall : 0.34, f1-score : 0.23} and accuracy of 0.94. In conclusion while we reached the accuracy threshold of 0.80 we were not happy with how we performed in the Drafted class. This model was a good start for us to build off of.  
![Model 1](https://github.com/DevonNoer/Project-4-NBA-Draft-Machine-Learning/assets/153139416/8c583aec-2463-4ca5-80ad-49b989a05f3c)
![Image](Images/Model_1_Graph.png)
## Model 2
The second model was also a neural network and it's features were(GP, mp, pts, Ortg, usg, TS_per, FTM, twoPM, TPM, drtg, bpm, oreb, dreb, ast, stl, blk, drafted). This model had three hidden layers, all the activation funcions were sigmoid, they had 20, 18, 16 neurons respectivly. The outer layer in this model also had an activation function of sigmoid. The classification report was Undrafted{precision : 0.99, recall : 0.97, f1-score : 0.98}, Drafted{precision : 0.00, recall : 0.00, f1-score : 0.00} and accuracy of 0.98. This model did not go how we were hoping because we ended up not getting anything for the drafted column, because our dataset is imbalanced we still achieved the accuarcy score of over 0.80. We learned that we are gonna have to make some big changes to increase our Drafted section, so we decided to look up ways to deal with an imbalanced dataset and use those ideas for our next models.
![Model 2](https://github.com/DevonNoer/Project-4-NBA-Draft-Machine-Learning/assets/153139416/f0b24c94-43d9-4736-986d-0f1746c09108)
![Image](Images/Model_2_Graph.png)
## Model 3
The third model was also a neural network and it's features were(conf, team, GP, mp, pts, Ortg, usg, TS_per, FTM, eFG, stops, FTA, twoPA, twoPM, TPM, TPA, drtg, bpm, oreb, dreb, treb, AST_per, TO_per, ast, stl, blk, Min_per, FT_per, twoP_per, ORB_per, DRB_per, blk_per, TP_per, yr, drafted). This model had four hidden layers, all the activation funcions were tanh, they had 90, 85, 80, and 75 neurons respectivly. The outer layer in this model also had an activation function of sigmoid. This model also used the class_weight to try and balance the data. The classification report was Undrafted{precision : 0.98, recall : 1.00, f1-score : 0.99}, Drafted{precision : 0.33, recall : 0.69, f1-score : 0.45} and accuracy of 0.96. Using class_weight was a helpful tool because our Drafted f1 score went up to a 0.45 which we thought was very good. We are still hoping to get the Drafted section up more if we can by changing weight, layers, neurons, and what ever else we can to improve our models.
![Model 3](https://github.com/DevonNoer/Project-4-NBA-Draft-Machine-Learning/assets/153139416/39f6ae24-d85d-4b44-8277-f9e108a9340e)
![Image](Images/Model_3_Graph.png)
## Model 4
The fourth model was also a neural network and it's features were(GP, mp, pts, Ortg, usg, TS_per, FTM, twoPM, TPM, drtg, bpm, oreb, dreb, ast, stl, blk, drafted). This model had three hidden layers, all the activation funcions were tanh, they had 60, 50, and 40 neurons respectivly. The outer layer in this model also had an activation function of sigmoid. This model also used the class_weight to try and balance the data. The classification report was Undrafted{precision : 0.99, recall : 0.95, f1-score : 0.97}, Drafted{precision : 0.24, recall : 0.64, f1-score : 0.35} and accuracy of 0.94. Using class_weight was a helpful tool because our Drafted f1 score went up to a 0.35 which was better than our orinal models but worse than our previous model. We are still hoping to get the Drafted section up more if we can by changing weight, layers, neurons, and what ever else we can to improve our models.
![Model 4](https://github.com/DevonNoer/Project-4-NBA-Draft-Machine-Learning/assets/153139416/e5a5c021-162a-486e-9151-6c88ad0f65c2)
![Image](Images/Model_4_Graph.png)
## Model 5
The fifth model was a neural network with the features(conf, GP, mp, pts, Ortg, usg, TS_per, FTM, FTA, twoPA, twoPM, TPM, TPA, drtg, bpm, oreb, dreb, ast, stl, blk, drafted). This model had two hidden layers, one with the activation function tanh and the other with relu. The output layer had the activation function of sigmoid. The hidden layers had 30 and 20 neurons, respectively. This was the classification report for this model.
Undrafted{precision: 0.99, recall: 0.96, f1-score: 0.97}, Drafted{precision: 0.26, recall: 0.65, f1-score: 0.37} and accuracy of 0.95. In conclusion, although the accuracy is high, we still want better precision for Drafted. We also used class weights to recognize drafted players better in the data, which helped improve the numbers from previous attempts.
![Image](Images/Model_5_Graph.png)

## Model 6
The sxith model was a neural network with the features(conf, MIN_per, pts, eFG, TS_per, FTA, twoPA, TPA, TPM, oreb, dreb, AST_per, TO_per, stl, blk, drafted). This model has three hidden layers, all relu, with 50, 40 and 20 neurons, respectively. Activation function of sigmoid on the outer layer. The following was the classification report: For the drafted category, the precision is 0.35, recall is 0.43, and F1-score is 0.39. For the undrafted category, the precision is 0.99, recall is 0.98, and F1-score is 0.98. The total accuracy of the model is 97% based on a total of 15,177 predictions. Using SMOTE (Synthetic Minority Over-sampling Technique) helped improve our model's performance by addressing the imbalance in the dataset, particularly enhancing the prediction accuracy, precision, recall, and F1-scores for the drafted category. By generating synthetic samples for the minority class (drafted), SMOTE enabled the model to better capture the patterns and characteristics of drafted players, leading to more accurate predictions and a more balanced evaluation of model performance across both drafted and undrafted categories.
![smote_model_6](https://github.com/DevonNoer/Project-4-NBA-Draft-Machine-Learning/assets/153139416/b7581fd7-97f5-4e07-8307-9ead9ca78823)

## Draft Preidictions
For the first attempt, we sourced data from a Reddit post, specifically an Excel file containing the 2023-2024 college season stats. This Excel file was converted to a CSV format for easier manipulation and integration with our model. We added the CSV file to our project files and altered the model's query to fit the chosen statistics from the new CSV to ensure compatibility. Despite the changes in the query, the original model still produced strong numbers. We used StandardScaler to scale the data, ensuring consistent input to the model. Predictions were generated and printed without retraining the model. Initially, the model predicted that only two players would be drafted: Alexandre Sarr (the #1 overall rated prospect and likely #1 pick) and Jarin Stevenson (who had dropped out of the draft a week ago, unbeknownst to us). Further reruns of the model showed predictions for Jarin Stevenson and Zach Edey (a center from Purdue) instead of Alexandre Sarr. The model's predictions were limited and did not capture a broader range of players, indicating a need for further refinement and potential data balancing.
<img width="339" alt="First_Attempt" src="https://github.com/DevonNoer/Project-4-NBA-Draft-Machine-Learning/assets/153139416/6f723466-485e-4187-8ea7-11d88b55f1c8">

In the second attempt, we used web scraping to gather data on the top 750 collegiate players from the previous season, sorted by scoring points per game. This larger dataset included many players who may never declare for the draft, but it provided a broader pool for our analysis. The stats from ESPN were averages, unlike the total stats in our original dataset, so adjustments were made accordingly. After preprocessing and scaling the data, we ran it through the model to identify draft-worthy players. The model's predictions highlighted the need for a more comprehensive and accurate database. In total, it identified 18 of 750 collegiate players that could be drafted. The following results are based on overall predictions two weeks before the NBA draft, verified by Googling the players and their draft stock.
<img width="524" alt="Draft_Predictions" src="https://github.com/DevonNoer/Project-4-NBA-Draft-Machine-Learning/assets/153139416/3c8ded4a-fde1-4cc4-808e-a03b07c86635">
<img width="387" alt="Second_Attempt" src="https://github.com/DevonNoer/Project-4-NBA-Draft-Machine-Learning/assets/153139416/7866239d-8f58-4937-a159-a8dac8e595b7">


To improve our draft prediction accuracy, we need access to a comprehensive database of information rather than relying on spreadsheets from Reddit or scraping websites for college data. A robust database containing all regular and advanced stats for both domestic and foreign players would provide a solid foundation. By continuously refining our model with high-quality data, we could significantly enhance its predictive power and potentially apply it in real-world scenarios.

# References
Used this website to find code to turn a classification report into a graph : https://medium.com/@Doug-Creates/plotting-scikit-learn-classification-report-for-analysis-0229447fe232 <br>
Google Slides : https://docs.google.com/presentation/d/1Ocn16RjP4gzTbt-MyaRb2DGWBDxWDFoIf-ICcamz_GI/edit#slide=id.p <br>
Dataset : https://www.kaggle.com/datasets/adityak2003/college-basketball-players-20092021/data?select=CollegeBasketballPlayers2009-2021.csv <br>
Imbalanced-Learn documentation : https://imbalanced-learn.org/stable/ <br>
Class Weights : https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
First Attempt Draft : https://www.reddit.com/r/NBA_Draft/comments/1c6ixes/2024_nba_draft_comprehensive_stat_spreadsheet/
Second Attempt Draft : https://www.espn.com/mens-college-basketball/stats/player
