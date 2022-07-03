# BasicAiReg

##  mix deps.get

# LSTM Based NER RNN
# Issues with the shapes
# Cant solve the input -> output shape issue::
17:18:03.243 [info]    StreamExecutor device (0): Host, Default Version
** (RuntimeError) branch operand 0 must match the shape of the only parameter of branch computation 0: got operand: (token[], f32[66,256], s64[128,100,1], f32[256,1024], f32[256,1024], /*index=5*/f32[256,1024], f32[256,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024], /*index=10*/f32[1024,1024], f32[1024], f32[1024], f32[1024], f32[1024], /*index=15*/s64[128,100,1], s64[128,100,1], s64[], s64[], f32[], /*index=20*/f32[9], f32[1024,9], f32[66,256], f32[1024], f32[1024], /*index=25*/f32[1024], f32[1024], f32[1024,1024], f32[1024,1024], f32[1024,1024], /*index=30*/f32[1024,1024], f32[256,1024], f32[1024,9], f32[9], s64[128,100,1]); computation: (p: (token[], f32[66,256], s64[128,100,1], f32[256,1024], f32[256,1024], /*index=5*/f32[256,1024], f32[256,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024], /*index=10*/f32[1024,1024], f32[1024], f32[1024], f32[1024], f32[1024], /*index=15*/s64[], f32[128,1,1024], f32[128,1,1024], f32[128,100,1024], f32[128,100,256], /*index=20*/f32[256,1024], f32[256,1024], f32[256,1024], f32[256,1024], f32[1024,1024], /*index=25*/f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024], f32[1024], /*index=30*/f32[1024], f32[1024], f32[1024,9], f32[9], s64[128,100,1])) -> (token[], f32[]).

##  mix run -e "Ner.run()"


# Basic from scratch
# https://keras.io/examples/nlp/text_classification_from_scratch/
##  mix run -e "NerB.run()"
# Results
14:50:14.056 [info]    StreamExecutor device (0): Host, Default Version
Epoch: 0, Batch: 0, accuracy: 0.2031250 loss: 0.0000000
Epoch: 1, Batch: 0, accuracy: 0.6015625 loss: 0.3883106
Epoch: 2, Batch: 0, accuracy: 0.6484375 loss: 0.3281645
Epoch: 3, Batch: 0, accuracy: 0.6484375 loss: 0.3025800
Epoch: 4, Batch: 0, accuracy: 0.6484375 loss: 0.2894246
Epoch: 5, Batch: 0, accuracy: 0.6484375 loss: 0.2812856
Epoch: 6, Batch: 0, accuracy: 0.6484375 loss: 0.2759556
Epoch: 7, Batch: 0, accuracy: 0.6484375 loss: 0.2718235
Epoch: 8, Batch: 0, accuracy: 0.6484375 loss: 0.2685038
Epoch: 9, Batch: 0, accuracy: 0.6484375 loss: 0.2659863
Epoch: 10, Batch: 0, accuracy: 0.6484375 loss: 0.2639405
Epoch: 11, Batch: 0, accuracy: 0.6484375 loss: 0.2620718
Epoch: 12, Batch: 0, accuracy: 0.6484375 loss: 0.2605137
Epoch: 13, Batch: 0, accuracy: 0.6484375 loss: 0.2592957
