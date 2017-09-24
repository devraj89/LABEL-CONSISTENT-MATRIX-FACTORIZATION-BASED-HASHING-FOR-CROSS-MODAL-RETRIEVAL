function [labels_new] = convert_labels_to_1_of_K(labels)

unq_lab = unique(labels);
labels_new = zeros(size(labels,1),length(unq_lab));
for i=1:size(labels_new,1)
    labels_new(i,labels(i)) = 1;
end