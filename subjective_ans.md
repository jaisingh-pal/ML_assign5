1] Are the results as expected? Why or why not?
Yes they are as expected. VGG one has only one block of concolution and pooling hence less accurate. Also data augmentation helps in generalizing and reduces overfitting.

2] Does data augmentation help? Why or why not?
Yes data augmentation reduces overfitting. Helps in 

3] Does it matter how many epochs you fine tune the model? Why or why not?
Yes, the number of epochs can be an important hyperparameter to fine-tune your model. The number of epochs determines how many times the model will iterate over the training data during training. If you train the model for too few epochs, it may not have sufficient time to learn the patterns in the data and may underfit. On the other hand, if you train the model for too many epochs, it may overfit to the training data and perform poorly on unseen data.
The optimal number of epochs depends on the complexity of the model and the amount of training data available. 
