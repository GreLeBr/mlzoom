# Prediction of a mushroom type

## Description
This project is an image classification of mushrooms based on the kitechenware competitions of DTC. 

## oups
I did not quite ended up finishing for the deadline so all the serving part is work in progress. 

## goal
The goal is simple and is to predict a mushroom based on its picture.  
I am using a dataset MO106 containing 106 classes of mushrooms available from https://keplab.mik.uni-pannon.hu/en/mo106eng  

I wanted to test the accuracy of different models and used similar approaches as shared from the kitchenware competion notebooks, namely using a quick fastai or using the SuperGradients Starter Notebook  both respectively available https://www.kaggle.com/code/miwojc/starter-notebook-with-fastai at  https://www.kaggle.com/code/harpdeci/supergradients-starter-notebook/notebook respectively.  

## ressources used
For fastai I bootstarted the identification using "convnext_tiny_in22k", I would not do justice to the better explanations available in the following introduction course and kaggle notebook https://course.fast.ai/Lessons/lesson3html and https://www.kaggle.com/code/jhoward/which-image-models-are-best/ but to quickly paraphrase it, convnext_tiny_in22k is a very good choice to maximize accuracy while keeping the model relatively lightweight and quick.  

I tested a few data augmentation operations to improve accuracy but settled on a rather simple approach of flipping, rotating at 90° and zooming in. 

From running a few runs I could achieve at best about 0.88 accuracy on my validation set. 


For the SuperGradients approach I mostly reused the code shared while adapating classes to my dataset. Again I invite any reader to head over the actual Kaggle notebook directly for better explanations as to what is used.   
Data augmentation is a bit of an art, at least the way I see it right now, and picking the right augmentation for the right problem can be key to improve performances and grind the last percent of accuracy.   

After running 10 epochs the model achieved 0.89 accuracy.  


# what's next
In both of these approaches the accuracy barely reach 90% which is not quite ideal when it comes to mushrooms and the decision of maybe eating it or not.

Classifying mushrooms is where an image classification can really shine, I am absolutely not knowledgeable in mushrooms so I will not have a good idea of where to go from there but I think running a decision trees checking a few key parameters of mushroom could help identify a mushroom in a situation where the prediction results is uncertain.    


The SuperGradients model ended up being almost 1go, the Fastai about 120mo and considering their accuracty being almost the same, I would go for the Fastai one in deployment.  
I had the idea of combining the two and averaging the prediction results to maybe get a better accuracty but I am not sure this would be very practical.  
I would need to run both prediction in parallel and adjust the result on a frontend.   

# so what is here

Well so far just the exploration notebook where I was trying to get BentoML to work on google colab with fastai (or pytorch), somehow neither works but it does on my laptop or on Saturn Cloud.   





	
	