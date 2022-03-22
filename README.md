# Stress Recognition in Thermal Videos 

This is the implementation of `Stress Recognition in Thermal Videos Using Bi-directional Long-Term Recurrent Convolutional Neural Networks(ICONIP2021)`.

To run the code, 

(1) data-preprocessing: video_data -> saved_frame -> remove_lastimgs-> stress_data_s8

-- step 1: change args_argument according to your aim, then run frame_extraction.py to extract frames from videos

-- step 2: run image_processing.py, it will remove first and last 4 frames for each "film", the aim is to avoid wrongly segmenting "flims".
		
-- step 3: remove images so that each folder has the same number of images
		
-- step 4: assign labels and create folders for each "film"
    
(2) python main_bilstm.py/main_lstm.py: Training our model with or without bi-directional training. You probably need to change the code according to your environment and Cuda version. You also can change code so that running other models from functions.py

(3) python generate_prediction.py: it will produce results on testset based on training checkpoints.
  
	 



