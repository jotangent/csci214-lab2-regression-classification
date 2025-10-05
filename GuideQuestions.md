# Regression Guide Questions

### a. How does your choice of window_size and stride affect model performance, generalization and stability of multi-step forecast
- Increasing window length for me should best be varying on the use case. I do not have a background in stocks so I created an experiment to check the MSEs of validation and training part of the model with varying window length. Note that I used vanilla model for both window length and stride variations.
- My method here is to analyze varying window lengths and stride separately, holding the other constant while turning the knob on the other, kind of like partial differentiation.
- Varying window lengths: Images below shows that increasing the window length indefinitely will result into increasing MSEs. Basing on the plots alone, I say that window length of 5 will yield the most generalized model in our case. There have been signs of overfitting on WL = 15, and WL = 25 where training MSE dipped while validation MSE increases.

![images/Training MSEs (window).png](<images/Training MSEs (window).png>)
![images/Val MSEs (window).png](<images/Val MSEs (window).png>)

- Varying strides: To avoid data leakage between training and validation data, and to have no overlapping between outputs, I decided to be strict that stride should be less than or equal to the number of outputs. There are other methods like having a robust way to combine overlapping outputs but that's another story, for now I'll stick with this idea. 
- Based on the images below, same with varying window length, increasing the stride seems to increase the MSEs for both training and validation phases.
- The only thing I can deduced from the plots below is to lower the stride and number of outputs as much as the business case allowed. This is my deduction because there's no signs of overfitting (low MSE for training high MSE for validation) in the plots below

![alt text](<images/Training MSEs (stride).png>)
![alt text](<images/Val MSEs (stride).png>)

### b. What are potential risks when recursively feeding predictions back as inputs for multi-step forecasting?
- Residuals are exponentially propagating as we feed recursive predictions to the model which is not realistic and will continue to expand the difference between predicted value and true values.
- Unrealistic input will yield unrealistic output. Acting on business problems using the unrealistic output will yield uncertain results.

### c. Compare the performance of the model with and without noise injection. Which method produced the most realistic forecast path? Explain both qualitative and quantitatively.
- I used two approaches in introducing noise to the model. First is to use gaussian noise injection in the input values during training; second is to have a residual storage I got from the validation data and then add them to the future data (uniformly picking from the list (with replacement)).
- Comparing the GNI to the vanilla in terms of metrics:
- Vanilla Metrics: train: 110.95781108311245 Val: 405.4354604085286
- GNI Metrics: train: 187.22298049926758 Val: 396.59190622965497
- Based from the metrics above, GNI performs better in terms of training (higher noise values -> check the code, I used knob=1.0). What i mean by better is, the training MSE increased and validation MSE decreased, which is a telltale sign that introducing the model gives a better generalization for unseen data.

### d. Explain the different parameters used for each noice injection method and how it might model predicting the future.
- The plot below is the future steps (recursive feeding) using model with GNI.

![alt text](<images/future plot - GNI.png>)

- Another method is to introduce noise by random sampling from the residual bank during validation noise. I only used this to create some variations in the plot for future prediction. I don't have metrics here since I just added the noise in the output of future steps.

![alt text](<images/future plot - output nosie.png>)

- Looking at the two plot, the residual bank method produced visually appealing residual plot BUT, since the purpose of forecasting is to deduce the future I still prefer the GNI. Yes it has smoother output but this is due to recursive feeding of smoothened output in each step.

### e. What does the error trend (RMSE vs predicted horizon) tell you about your model's ability to extrapolate.
- I actually don't understand the question. what should I compare RMSE to? Which RMSE (vanilla, gni)?


# Classification Guide Questions

Two methods were explored in engineering labels from the time-series dataset.
1. **Triple barrier method**, which I modified into binary outputs ignoring the timeout parameter and assumed it to be just 'SELL' label for conservative approach (I was short on time to experiment on F1 scores and modelling for multiclass).
2. **Moving average method**, which separates classes into 'increasing' or 'decreasing'.

### a. Which engineered features contributed most to a higher F1 score and why do you think this is the case? Since you defined how to compute for a given label, explain this in terms of how a neural network attempts to learn your method.
- The resulting f1 for both during validation phase are: Triple Barrier (modified into binary classes) F1 = 0.827; Moving average F1 = 0.826
- Both engineered method yielded almost balanced data for both training and validation sets. (below are the training and validation distribution)
- Both methods yielded almost identical engineering (despite having different appraoch), which made the model's approximation almost identitical for both.

TRIPLE-BARRIER<br>
train counts: Counter({1: 170, 0: 130})<br>
val counts:   Counter({1: 70, 0: 58})

MOVING-AVERAGE<br>
train counts: Counter({1: 172, 0: 128})<br>
val counts:   Counter({1: 78, 0: 50})


### b. How did your labeling scheme affect class balance and therefore F1 performance? Answer this with quantitative evidence.
- Both labeling scheme yielded fairly balanced data in terms of training. Since F1 score is the harmonic mean of precision and recall, it accounts for both false positives and negatives. I didnt have a problem in optimizing for F1 performance due to a balanced engineering. F1 should be improved using hyper parameter tuning, I just didn't have enough time to do some optimization. 

### c. If F1 was low for one or more classes, which additional features or transformations could you add to improve it?
- I'd only answer this on theoretical sense, but feature transformations like z-score scaling instead of the raw values might improve F1 score.
- Regarding the triple barrier method, I put in a knob (some sort of hyperparameter) to adjust the effects of the input data in calculating the barrier thresholds.
- Another approach is to account for seasonality in choosing window_length, horizon (for labeling), and stride.

### d. What patterns do you observe in false positives vs false negatives when F1 is low, and which feature adjustments could reduce these errors?
- Since F1 is a harmonic mean for precision and recall, it will be low if there is an increase in either False positives and negatives or both. Same with my answer with question c. introduce feature transformations especially in the training input (X). 