from fastai.vision import load_learner, open_image, Path
import sys

skates_type = sys.argv[1]

# Create a path object
path = Path('./')

# The classifier was not trained on this image
img = open_image('./' + skates_type + '.jpeg')

learn = load_learner(path)

pred_class, _, __ = learn.predict(img)

print(pred_class)
