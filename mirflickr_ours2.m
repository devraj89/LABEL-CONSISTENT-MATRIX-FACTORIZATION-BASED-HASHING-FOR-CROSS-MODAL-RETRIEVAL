clear all; clc; close all;
warning off;
addpath(genpath('\markSchmidt\'));

%% Parameter Setting
globalBits = [16,32,64,128];
datasets = {'mirflickr'};

N = 22500;

kernelSamps = [500, 500];                            % sampling size for kernel logistic regression

dtN = length(datasets);
recallLevelStep = 0.05;

% Model parameters
Model = {};
Model.alpha = 1e-2;

%% SePH
for di = 1 : dtN
    clearvars -except globalBits datasets dtN di recallLevelStep fid Model kernelSamps N;
    
    load('datasets\reduced_mirflickr.mat');    
    
    % For the MirFlickr dataset
    % consider 2500(10% randomly sampled pairs) as the query set and the
    % rest as the training set
    t = randperm(size(labels,1));
    I_te = image_feat(t(1:2500),:);
    T_te = text_feat(t(1:2500),:);
    L_te = labels(t(1:2500),:);
    I_tr = image_feat(t(2501:end),:);
    T_tr = text_feat(t(2501:end),:);
    L_tr = labels(t(2501:end),:);
    sampleInds = 1:size(I_tr,1);
    clear t image_feat text_feat labels
    
    sampleInds = sampleInds(1:N);
    
    v = 2;
    viewsName = {'Image', 'Text'};
    
    RetrXs = cell(1, v);                            % Retrieval Set
    RetrXs{1} = I_tr;
    RetrXs{2} = T_tr;
    RetrXs{3} = L_tr;
    
    queryXs = cell(1, v);                           % Query Set
    queryXs{1} = I_te;
    queryXs{2} = T_te;
    queryXs{3} = L_te;
    clear I_tr T_tr I_te T_te;
    
    % Feature Pretreatment
    for i = 1 : v
        meanV = mean(RetrXs{i}, 1);
        RetrXs{i} = bsxfun(@minus, RetrXs{i}, meanV);
        queryXs{i} = bsxfun(@minus, queryXs{i}, meanV);
    end
    
    trainNum = length(sampleInds);                  % Training Set
    trainXs = cell(1, v);
    trainXs{1} = RetrXs{1}(sampleInds, :);
    trainXs{2} = RetrXs{2}(sampleInds, :);
    trainXs{3} = RetrXs{3}(sampleInds, :);
    
    % Calculation of P for supervised learning (normalized cosine similarity)
    tr_labels = L_tr(sampleInds, :);
    
    % Training & Testing
    bitN = length(globalBits);
    bits = globalBits;
    
    queryNum = size(L_te, 1);
    
    runtimes = 10;                                                  % 10 runs
    mAPs = zeros(bitN, v, runtimes, 1);
    trainMAPs = zeros(bitN, runtimes);
    
    for bi = 1 : bitN
        bit = bits(bi);
        
        for ri = 1 : runtimes
            
            %%
%             clc
            tic
            % This can be done in parallel ? Why ? Because we will not be
            % using the U1 U2 and U3 to generate the V's
            % select the optimum parameters please
            if bit==16
                lambdas = [1 1 10]; gamma = 0.1; alphas = [1 1];
            elseif bit==32
                lambdas = [0.1 1 100]; gamma = 0.1; alphas = [1 1];
            elseif bit==64
                lambdas = [0.1 1 100]; gamma = 0.1; alphas = [1 1];
            elseif bit==128
                lambdas = [1 1 10]; gamma = 0.1; alphas = [1 1];
            else
                lambdas = [1 1 10]; gamma = 0.1; alphas = [1 1];
            end
            
            % select a subset of the data to generate the hash codes
            Total_size = N;
            A = zeros(Total_size,bit); B = zeros(Total_size,bit);
            subset_size = 1000;
            step = 1;
            steps = length(1:subset_size:Total_size);
            trEv = [];
            h = waitbar(0,'Please wait...');
            for counter = 1:subset_size:Total_size
                t1 = counter;
                t2 = t1+(subset_size-1);
                if t2>N
                    t2 = N;
                end
                train_data{1} = trainXs{1}(t1:t2,:); train_data{2} = trainXs{2}(t1:t2,:); train_data{3} = trainXs{3}(t1:t2,:);
                if step==1
                    [U1, U2, U3, ~, ~, ~, W1, W2] = solveUCMFH_devraj3(train_data{1}, train_data{2}, train_data{3}, lambdas, gamma, alphas, bit);
                    rho = 0.1;
                    [U1, U2, U3, V1, V2, V3, W1, W2] = solveUCMFH_devraj3_propagate(train_data{1}, train_data{2}, train_data{3}, lambdas, gamma, alphas, bit, U1,U2,U3,W1,W2, rho);
                elseif step>1                    
                    [U1, U2, U3, V1, V2, V3, W1, W2] = solveUCMFH_devraj3_propagate(train_data{1}, train_data{2}, train_data{3}, lambdas, gamma, alphas, bit, U1,U2,U3,W1,W2, rho);
                end
                % hash codes are as follows
                X = sign((V3).'); Y = sign((V3).');
                A(t1:t2,:) = X; B(t1:t2,:) = Y;
                
% % % %                 % Evaluating the Quality of Learnt Hash Codes for Training Set
% % % %                 trEv(step) = trainEval2(tr_labels(t1:t2,:), X, Y);                
% % % %                 fprintf('Step %d Manifold Evaluation MAP [%.4f]\r', step, trEv(step));
                waitbar(step/steps,h)
                step = step + 1;
            end
            close(h)
%             mean(trEv)
            
% % % %             % Evaluating the Quality of Learnt Hash Codes for Training Set
% % % %             trEv = trainEval2(tr_labels(1:Total_size,:), A, B);
% % % %             fprintf('Manifold Evaluation MAP [%.4f]\r', trEv);
            toc
            
            
            %%
            
            % Now since the data is too large - I will not be able to deal
            % with it so I will really select a small subset of the data to
            % learn the projection vectors using the logistic regression
            Numb_of_Samp = 5000;
            train_data = [];
            train_data{1} = trainXs{1}(1:Numb_of_Samp,:);
            train_data{2} = trainXs{2}(1:Numb_of_Samp,:);
            trainNum = Numb_of_Samp;
            
            % RBF Kernel
            z = train_data{1} * train_data{1}';
            z = repmat(diag(z), 1, trainNum)  + repmat(diag(z)', trainNum, 1) - 2 * z;
            k1 = {};
            k1.type = 0;
            k1.param = mean(z(:));                                  %  $\sigma^2$ for RBF kernel in image view
            
            z = train_data{2} * train_data{2}';
            z = repmat(diag(z), 1, trainNum)  + repmat(diag(z)', trainNum, 1) - 2 * z;
            k2 = {};
            k2.type = 0;
            k2.param = mean(z(:));                                  %  $\sigma^2$ for RBF kernel in text view
            
            % Kernel Logistic Regression (KLR)£¬Developed by Mark Schimidt
            for si = 1 : 1
                kernelSampleNum = kernelSamps(di);
                if si == 1 && kernelSampleNum > trainNum
                    break;
                elseif si == 2 && kernelSampleNum > trainNum / 2
                    break;
                end
                
                sampleType = 'Random';
                if si == 1
                    % Random Sampling for Learning KLR
                    kernelSamples = sort(randperm(trainNum, kernelSampleNum));
                    kernelXs{1} = train_data{1}(kernelSamples, :);
                    kernelXs{2} = train_data{2}(kernelSamples, :);
                else
                    sampleType = 'Kmeans';
                    % Kmeans Sampling for Learning KLR
                    opts = statset('Display', 'off', 'MaxIter', 100);
                    [INX, C] = kmeans(train_data{1}, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
                    kernelXs{1} = C;
                    
                    [INX, C] = kmeans(train_data{2}, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
                    kernelXs{2} = C;
                end
                
                % Kernel Matrices
                K01 = kernelMatrix(kernelXs{1}, kernelXs{1}, k1);
                K02 = kernelMatrix(kernelXs{2}, kernelXs{2}, k2);
                trainK1 = kernelMatrix(train_data{1}, kernelXs{1}, k1);
                trainK2 = kernelMatrix(train_data{2}, kernelXs{2}, k2);                
                RetrK1 = kernelMatrix(RetrXs{1}, kernelXs{1}, k1);
                RetrK2 = kernelMatrix(RetrXs{2}, kernelXs{2}, k2);
                queryK1 = kernelMatrix(queryXs{1}, kernelXs{1}, k1);
                queryK2 = kernelMatrix(queryXs{2}, kernelXs{2}, k2);
                
                % Hash Codes for Retrieval Set and Query Set                
                B1 = zeros(size(L_tr, 1), bit);                             % Unique Hash Codes for Both Views of Retrieval Set
                B21 = zeros(queryNum, bit);                                 % Hash Codes for Image View of Query Set
                B22 = zeros(queryNum, bit);                                 % Hash Codes for Text View of Query Set
                
                options.Display = 'final';
                C = 0.01;                                                   % Weight for Regularization. 1e-2 is Good Enough.
                
                % KLR for Each Bit
                for b = 1 : bit                    
                    tH = A(1:Numb_of_Samp, b);                    
                    % View 1 (Image View)
                    funObj = @(u)LogisticLoss(u, trainK1, tH);
                    w = minFunc(@penalizedKernelL2, zeros(size(K01, 1),1), options, K01, funObj, C);
                    B21(:, b) = sign(queryK1 * w);
                    z11 = 1 ./ (1 + exp(-RetrK1 * w));                                     % P(pos | V_1)
                    z10 = 1 - z11; 
                    
                    tH = B(1:Numb_of_Samp, b);                    
                    % View 2 (Text View)
                    funObj = @(u)LogisticLoss(u, trainK2, tH);
                    w = minFunc(@penalizedKernelL2, zeros(size(K02, 1),1), options, K02, funObj, C);
                    B22(:, b) = sign(queryK2 * w);
                    z21 = 1 ./ (1 + exp(-RetrK2 * w));                                     % P(pos | V_2)
                    z20 = 1 - z21;
                    
                    % Retrieval Set (Combining)
                    B1(:, b) = sign(z11 .* z21 - z10 .* z20);
                end
                
                B1 = bitCompact(sign(A) >= 0);
%                 B1 = bitCompact(sign(B1) >= 0);
                B21 = bitCompact(sign(B21) >= 0);
                B22 = bitCompact(sign(B22) >= 0);
                
                % Evaluation
                vi = 1;
                fprintf('Computing Map@50 for Text-to-Image\r');  
                hammingM = 1-double(HammingDist(B21, B1))';
                mAPValue = map_at_50(hammingM,L_tr,L_te);
                mAPs(bi, vi, ri, si) = mAPValue;
                fprintf('%s Bit %d Runtime %d Sampling Type [%s] Sampling Num [%d], %s query %s: MAP [%.6f]\r', ...,
                    datasets{di}, bit, ri, sampleType, kernelSampleNum, viewsName{1}, viewsName{2}, mAPValue);
                
                vi = 2;
                fprintf('Computing Map@50 for Image-to-Text\r');
                hammingM = 1-double(HammingDist(B22, B1))';
                mAPValue = map_at_50(hammingM,L_tr,L_te);
                mAPs(bi, vi, ri, si) = mAPValue;
                fprintf('%s Bit %d Runtime %d Sampling Type [%s] Sampling Num [%d], %s query %s: MAP [%.6f]\r', ...,
                    datasets{di}, bit, ri, sampleType, kernelSampleNum, viewsName{2}, viewsName{1}, mAPValue);
            end
            
        end
    end
end

clc
X = mAPs(1,1,:); Y = mAPs(1,2,:);
fprintf('%s Bit %d %s query %s: MAP [%.6f]\r', ...,
    datasets{di}, bit, viewsName{1}, viewsName{2}, mean(X));
fprintf('%s Bit %d %s query %s: MAP [%.6f]\r', ...,
    datasets{di}, bit, viewsName{2}, viewsName{1}, mean(Y));