# Given tye of task(s) and modalities, we will likely train and operate via colab (or other cloud)
# we therefore won't be able to generate a significant amount of labelled data (images)
# and push them via github (over limit) or drive, is probably better imho to structure as: 
#       we import (online) the FEN notated data, and either use a Generator class (memory efficient), containing utilities 
#       to generate images from FEN (that is imported in known location) or otherwise by batch generate them and save
#       them in a defined location. Choice depends on how scarce memory is (Generators are more memory efficient)
# 
# 
# 
# 
   
