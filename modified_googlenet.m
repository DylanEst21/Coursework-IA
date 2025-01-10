
net = googlenet;
deepNetworkDesigner  

%{
We loaded googlenet and now we use this tool to import it from the Workspace to modify it:
- We delete the last three layers: loss3-classifier, prob, and output.
- We replaced them with a new Fully Connected (new_fc) layer with
OutputSize = 11, a Softmax layer (new_softmax) and a Classification Output
layer (new_classoutput).

Finaly, we exported the network "lgraph_1" to the Workspace and saved it
to a file.
%}


save('modified_googlenet_layers.mat', 'lgraph_1');
