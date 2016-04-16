#Audio / Video Synchronization
Brian Pugh & Nicholas Moschopoulos <br />
Berkeley Computer Science 280

###Abstract

This project explores techniques for recombining desynchronized audio and video tracks.

###FaceSync

We attempt to reconstruct the algorithm from the Slaney, Covell paper (2001). It implements an optimal linear algorithm which calculates the correlation between Mel Frequency Cepstrum Coefficients and image face pixels over time. 

###Left/Right Video Alignment

We split a video segment into the left and right half of the image. One half is then offset in time by anywhere from 0 to 10 frames. A Siamese Convolutional Neural Network is trained to predict the alignment of the video. The truths are 1 for aligned and 0 for unaligned.

#### Dataset

The Left/Right Video Alignment dataset contains X sequences of ten frames each. If the label for a sequence is 1, then the left and right sides of the video are temporally offset. We do this by adding some temporal offset less than ten to find the second half. For example, the left half could be frames 8-17 while the right half is 10-19. If the label is 1, there is some non-zero offset. If the label is 0, there is no offset. <br />

We remove a ten pixel bar in the middle of the video frame in order to prevent the video from simply learning to match the brightnesses or gradients of neighboring pixels. <br />

In creation, we create one aligned and one unaligned pair for each twenty frame source sequence.