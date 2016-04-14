#Audio / Video Synchronization
##Brian Pugh & Nicholas Moschopoulos
##Berkeley Computer Science 280

This project explores techniques for recombining desynchronized audio and video tracks.

###FaceSync

We attempt to reconstruct the algorithm from the Slaney, Covell paper (2001). It implements an optimal linear algorithm which calculates the correlation between Mel Frequency Cepstrum Coefficients and image face pixels over time. 

###Left/Right Video Synchronization

We split a video segment into the left and right half of the image. One half is then offset in time by anywhere from 0 to X frames. A Siamese Convolutional Neural Network is then used to predict the synchronicity of the video. The truths are 1 for synchronized and 0 for desynchronized.
