
net = alexnet;
deepNetworkDesigner  

%{
We loaded alexnet and now we use this tool to import it from the Workspace to modify it:
- We delete the last three layers:  fc8, prob (softmax), and output (classification).
- We replaced them with a new Fully Connected (new_fc8) layer with
OutputSize = 11, a Softmax layer (new_softmax) and a Classification Output
layer (new_classoutput).

Finaly, we exported the network "layers_1" to the Workspace and saved it
to a file.
%}


save('modified_alexnet_layers.mat', 'layers_1');
