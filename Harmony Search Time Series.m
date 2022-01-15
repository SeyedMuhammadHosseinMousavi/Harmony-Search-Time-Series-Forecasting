%% Harmony Search Time Series Forecasting - Created in 15 Jan 2022 by Seyed Muhammad Hossein Mousavi
% This code uses Harmony Search evolutionary algorithm in order to forecast
% desired steps into the future based on input vector of time series data.
% You can load your data vector and play with parameters based on your
% data and system.
% ------------------------------------------------ 
% Feel free to contact me if you find any problem using the code: 
% Author: SeyedMuhammadHosseinMousavi
% My Email: mosavi.a.i.buali@gmail.com 
% My Google Scholar: https://scholar.google.com/citations?user=PtvQvAQAAAAJ&hl=en 
% My GitHub: https://github.com/SeyedMuhammadHosseinMousavi?tab=repositories 
% My ORCID: https://orcid.org/0000-0001-6906-2152 
% My Scopus: https://www.scopus.com/authid/detail.uri?authorId=57193122985 
% My MathWorks: https://www.mathworks.com/matlabcentral/profile/authors/9763916#
% my RG: https://www.researchgate.net/profile/Seyed-Mousavi-17
% ------------------------------------------------ 
% Related Paper For Cite:
% Mousavi, Seyed Muhammad Hossein, S. Younes MiriNezhad, and Mir Hossein Dezfoulian. "Galaxy gravity optimization (GGO) an algorithm for optimization, inspired by comets life cycle." 2017 Artificial Intelligence and Signal Processing Conference (AISP). IEEE, 2017.
% Hope it help you, enjoy the code and wish me luck :)

%% Clearing Things
clc
clear
close all
warning ('off');

%% Data Load and Preparation
Data=load('CovidTimeSeries');
Data=Data.CovidTimeSeries';
Data=Data(1:400);

%% K is Number of Desired Steps into the Future 
K = 100;

%% Basic Forecast
nstep = 80;
sys = nlarx(Data,nstep);
opt = forecastOptions('InitialCondition','e');
[Future,ForecastMSE] = forecast(sys,Data,K,opt);
datsize=size(Data);
datsize=datsize(1,1);
ylbl=datsize+K;
t = linspace(datsize,ylbl,length(Future));
% Creating Inputs and Targets
Delays = [1];
[Inputs, Targets] = MakeTheTimeSeries(Future',Delays);
data.Inputs=Inputs;
data.Targets=Targets;
% Making Data
Inputs=data.Inputs';
Targets=data.Targets';
Targets=Targets(:,1);
nSample=size(Inputs,1);
% Shuffle Data
% S=randperm(nSample);
% Inputs=Inputs(S,:);
% Targets=Targets(S,:);
% Train Data
pTrain=1.0;
nTrain=round(pTrain*nSample);
TrainInputs=Inputs(1:nTrain,:);
TrainTargets=Targets(1:nTrain,:);
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end,:);
% Making Final Data Struct
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
data.TestInputs=TestInputs;
data.TestTargets=TestTargets;

%% Basic Fuzzy Model Creation 
ClusNum=8;      % Number of Clusters in FCM
fis=GenerateFuzzy(data,ClusNum);

%% Tarining Harmony Search Algorithm
HarmonySearchFis = HarmonySearchFCN(fis,data); 

%% Plot Fuzzy Harmony Search Time Series Train Result
% Train Output Extraction
TrTar=data.TrainTargets;
TrainOutputs=evalfis(data.TrainInputs,HarmonySearchFis);
% Train calculation
Errors=data.TrainTargets-TrainOutputs;
r0 = -1 ;
r1 = +1 ;
range = max(Errors) - min(Errors);
Errors = (Errors - min(Errors)) / range;
range2 = r1-r0;
Errors = (Errors * range2) + r0;
MSE=mean(Errors.^2);
RMSE=sqrt(MSE);  
error_mean=mean(Errors);
error_std=std(Errors);
% Train Target and Output
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,1,1);
plot(data.TrainTargets,'--c','LineWidth',1);
hold on;
plot(TrainOutputs,'k','LineWidth',1);
legend('Target','Output');
title('Harmony Search Training Part');
xlabel('Sample Index');
grid on;
% Train MSE
subplot(3,1,2);
plot(Errors,':k','LineWidth',2);
legend('Harmony Search Training Error');
title(['Train MSE =     ' num2str(MSE) '  ,     Train RMSE =     ' num2str(RMSE)]);
grid on;
% Train Error
subplot(3,1,3);
h=histfit(Errors, 50,'kernel');
h(1).FaceColor = [.8 .4 0.3];
h(2).Color = 'b';
title(['Train Error Mean =   ' num2str(error_mean) '  ,   Train Error STD =   ' num2str(error_std)]);
%% Regression Line
figure;
[population2,gof] = fit(TrainTargets,TrainOutputs,'poly3');
plot(TrainTargets,TrainOutputs,'o',...
    'LineWidth',2,...
    'MarkerSize',4,...
    'Color',[0.3,0.2,0.9]);
    title(['Train - R =  ' num2str(abs(1-gof.rmse))]);
        xlabel('Train Target');
    ylabel('Train Output');   
hold on
plot(population2,'r-','predobs');
    xlabel('Train Target');
    ylabel('Train Output');   
%% Harmony Search Forecast Plot and Compare
K2=size(TrainOutputs);
K2 = K2(1,1);
datsize=size(Data);
datsize=datsize(1,1);
ylbl=datsize+K2;
t2 = linspace(datsize,ylbl,length(TrainOutputs));
% Plot
figure;
set(gcf, 'Position',  [50, 50, 1200, 300])
plot(Data,'-',...
    'LineWidth',1,...
    'MarkerSize',5,...
    'Color',[0,0,0]);
hold on;
plot(t2,TrainOutputs,'-',...
    'LineWidth',1,...
    'MarkerSize',5,...
    'MarkerEdgeColor','r',...
    'Color',[0.9,0,0]);
title('Harmony Search Forecast')
xlabel('Days','FontSize',12,...
       'FontWeight','bold','Color','r');
ylabel('Cases','FontSize',10,...
       'FontWeight','bold','Color','r');
hold on;
plot(t,Future,'-',...
    'LineWidth',1,...
    'MarkerSize',5,...
    'MarkerEdgeColor','r',...
    'Color',[0,0.9,0]);
legend({'Measured','Harmony Search Forecast','Basic Forecast'});

%% Harmony Search Time Series Forecasting Performance Statistics
fprintf('Harmony Search MSE Is =  %0.4f.\n',MSE)
fprintf('Harmony Search RMSE Is =  %0.4f.\n',RMSE)
fprintf('Harmony Search Train Error Mean Is =  %0.4f.\n',error_mean)
fprintf('Harmony Search Train Error STD Is =  %0.4f.\n',error_std)

