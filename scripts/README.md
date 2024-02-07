# Evaluate

You can evaluate the framework and generate videos of the gestures by executing the following commands in the main project folder: \
``` bash eval_AQ-GT.sh ``` for the evaluation of the AQ-GT model. \
``` bash eval_AQ-GT-A.sh ``` for the evaluation of the AQ-GT-A model. 

the generated videos can be found in the ```test_full``` folder

Currently, the framework only takes pre-created lmdb data files as input and has no direct interface to create realtime gestures from videos

If you want to create a new lmdb data file, please refer to the "new-youtube-gesture-dataset" folder. \

##  Synthesizing using modifiers

The frameworkd allows to use specific modifiers, which can be changed to influence the gesture generation. 
Currently these are hardcoded in the code and can be found in the ``` synthesize_full.py```  file, inside of the ```main```  function

These modifiers are implemented as tuple consisting of 

``` name of the output file, text modifier, audio modifier, annotation_modifiers ```

for the audio and text modifier we currently support removing the specific input modality, by setting the modifier to None, like so:

No text modality: ``` ["no text", None, 0, []]``` 

No audio modality: ``` ["no audio", 0, None, []]``` 

The annotation modifier are more powerful, allowing to remove, alter, overwrite, or even add parts of the annotation.
These annotation modifiers consist of a list of three Elements in a tuple:  \
Like so: ``` (Index of the annotation to be modified, Source values, Destination Values) ``` 

The index is the Element that should be changed. The index for the annotation schema is as follows:  

entity_map       = 0 ; with a value between 1 and 18 \
occurence_map    = 1 ; with a value between 0 and 1\
left phase       = 2 ; with a value between 1 and 5\
right phase      = 3 ; with a value between 1 and 5\
left phrase         = 4 ; with a value between 1 and 8\
right phrase         = 5 ; with a value between 1 and 8\
left position       = 6 ; with a value between 1 and 13\
right position       = 7 ; with a value between 1 and 13\
speech position       = 8 ; with a value between 1 and 13\
left hand_shape     = 9 ; with a value between 1 and 20\
right hand_shape     = 10 ; with a value between 1 and 20\
left wrist distance = 11 ; with a value between 1 and 6\
right wrist distance = 12 ; with a value between 1 and 6\
left extend         = 13 ; with a value between 1 and 4\
right extend         = 14 ; with a value between 1 and 4\
left practice       = 15 ; with a value between 1 and 14\
right practice       = 16 ; with a value between 1 and 14\

Please refer to the ```AQGT/scripts/data_loader/data_preprocessor.py``` file, to find out the clear name label of each annotation. \
``` Please also note, that we add +1 to each of the clear name values to have an empty padding value ``` 

After giving the index (or multiple indices) of the annotation as the first element, we can change the parts of the input with the "Source" and "Destination Values".
We can give multiple entries as the source values and a single destination value. For the specified index, all values with the source values are then changed to the destination value.

For example. To modify the generation to always have large instead of small gestures, we could add:
```["small to wide extend gestures", 0, 0, [((13, 14), (1, 2), 3)]]``` 

The annotation schema ```[((13, 14), (1, 2), 3)]```  is here as follows: \
(13,14) are the index for the left and right extend of the hands. \
(1,2) are the values for small(1) and medium(2) extend. \
3 is the value for the large extend. 

There is one exception. If the source values is specified as None, then the value is treated as a wildcard and any values in the specified index are overwritten with the destination value:
For example: \
```["always do strokes", 0, 0, [((2, 3), (None,), 1), ]]``` would always set strokes(1) for the left(2) and right(3) phase.



