clear all
w = 6; % Set w
r = 10; % Set r
d=-8; % Set initial d to -8
P = 2000;
xin = linspace(-15,25,P); % Generate 2000 values in x direction
yin = linspace(-15,15,P); % Generate 2000 values in y-direction
[x, y] = meshgrid(xin,yin); % Create grid of coordinates for boundary drawing
rng(123) % Set seed for training data
radius1 = sqrt(((r+w/2)^2-(r-w/2)^2)*rand(1,1000)+(r-w/2)^2); % Set radius
angle1 = pi*rand(1,1000); % Set angle
x1 = radius1.*cos(angle1); % Calculate x coordinates of class 1
y1 = radius1.*sin(angle1); % Calculate y coordinates of class 1
radius2 = sqrt(((r+w/2)^2-(r-w/2)^2)*rand(1,1000)+(r-w/2)^2); % Set radius
angle2 = pi + pi*rand(1,1000); % Set angle
x2 = radius2.*cos(angle2) + r; % Calculate x coordinates of class 2
y2 = radius2.*sin(angle2) - d; % Calculate y coordinates of class 2
rng(111) % Set seed for testing data
radius3 = sqrt(((r+w/2)^2-(r-w/2)^2)*rand(1,500)+(r-w/2)^2);
angle3 = pi*rand(1,500);
x3 = radius3.*cos(angle3);
y3 = radius3.*sin(angle3);
radius4 = sqrt(((r+w/2)^2-(r-w/2)^2)*rand(1,500)+(r-w/2)^2);
angle4 = pi + pi*rand(1,500);
x4 = radius4.*cos(angle4) + r;
y4 = radius4.*sin(angle4) - d;
ydes = [zeros(1,500) ones(1,500)];
dat = [x1 x2; y1 y2;zeros(1,1000) ones(1,1000)]; % Create training data
rng(22)
dat = dat(:,randperm(length(dat))); % Shuffle training data
%%
net1 = patternnet(20);
net1.trainParam.lr = 0.05;
net1.trainFcn = 'traingd';
net1.performFcn = 'mse';
%net1.divideFcn = 'dividetrain';
net1 = configure(net1,[dat(1,:); dat(2,:)],dat(3,:));
net1.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net1 = init(net1);
net1.iw{1,1}=iw_1;
net1.lw{2,1}=lw_1';
net1.b{1}= zeros(numofneurons,1);
[net1, tr1] = train(net1,[dat(1,:); dat(2,:)],dat(3,:));
yout1 = sim(net1,[x3 x4;y3 y4])>0.5;
%%
net2 = patternnet(20);
net2.trainParam.lr = 0.05;
net2.trainFcn = 'traingdm';
net2.performFcn = 'mse';
%net2.divideFcn = 'dividetrain';
net2 = configure(net2,[dat(1,:); dat(2,:)],dat(3,:));
net2.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net2 = init(net2);
net2.iw{1,1}=iw_1;
net2.lw{2,1}=lw_1';
net2.b{1}= zeros(numofneurons,1);
[net2, tr2] = train(net2,[dat(1,:); dat(2,:)],dat(3,:));
yout2 = sim(net2,[x3 x4;y3 y4])>0.5;
%%
net3 = patternnet(20);
net3.trainParam.lr = 0.05;
net3.trainFcn = 'trainlm';
net3.performFcn = 'mse';
%net3.divideFcn = 'dividetrain';
net3 = configure(net3,[dat(1,:); dat(2,:)],dat(3,:));
net3.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net3 = init(net3);
net3.iw{1,1}=iw_1;
net3.lw{2,1}=lw_1';
net3.b{1}= zeros(numofneurons,1);
[net3, tr3] = train(net3,[dat(1,:); dat(2,:)],dat(3,:));
yout3 = sim(net3,[x3 x4;y3 y4])>0.5;
%%
net4 = patternnet(20);
net4.trainParam.lr = 0.5;
net4.trainFcn = 'traingdm';
net4.performFcn = 'mse';
%net4.divideFcn = 'dividetrain';
net4 = configure(net4,[dat(1,:); dat(2,:)],dat(3,:));
net4.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net4 = init(net4);
net4.iw{1,1}=iw_1;
net4.lw{2,1}=lw_1';
net4.b{1}= zeros(numofneurons,1);
[net4, tr4] = train(net4,[dat(1,:); dat(2,:)],dat(3,:));
yout4 = sim(net4,[x3 x4;y3 y4])>0.5;
%%
net5 = patternnet(20);
net5.trainParam.lr = 0.005;
net5.trainFcn = 'traingdm';
net5.performFcn = 'mse';
%net5.divideFcn = 'dividetrain';
net5 = configure(net5,[dat(1,:); dat(2,:)],dat(3,:));
net5.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net5 = init(net5);
net5.iw{1,1}=iw_1;
net5.lw{2,1}=lw_1';
net5.b{1}= zeros(numofneurons,1);
[net5, tr5] = train(net5,[dat(1,:); dat(2,:)],dat(3,:));
yout5 = sim(net5,[x3 x4;y3 y4])>0.5;
%%
net6 = patternnet(5);
net6.trainParam.lr = 0.05;
net6.trainFcn = 'traingdm';
net6.performFcn = 'mse';
%net6.divideFcn = 'dividetrain';
net6 = configure(net6,[dat(1,:); dat(2,:)],dat(3,:));
net6.trainParam.epochs = 5000;
numofneurons= 5;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net6 = init(net6);
net6.iw{1,1}=iw_1;
net6.lw{2,1}=lw_1';
net6.b{1}= zeros(numofneurons,1);
[net6, tr6] = train(net6,[dat(1,:); dat(2,:)],dat(3,:));
yout6 = sim(net6,[x3 x4;y3 y4])>0.5;
%%
net7 = patternnet(40);
net7.trainParam.lr = 0.05;
net7.trainFcn = 'traingdm';
net7.performFcn = 'mse';
%net7.divideFcn = 'dividetrain';
net7 = configure(net7,[dat(1,:); dat(2,:)],dat(3,:));
net7.trainParam.epochs = 5000;
numofneurons= 40;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net7 = init(net7);
net7.iw{1,1}=iw_1;
net7.lw{2,1}=lw_1';
net7.b{1}= zeros(numofneurons,1);
[net7, tr7] = train(net7,[dat(1,:); dat(2,:)],dat(3,:));
yout7 = sim(net7,[x3 x4;y3 y4])>0.5;
%%
d=-4;
y2 = radius2.*sin(angle2) - d;
y5 = radius4.*sin(angle4) - d;
dat = [x1 x2; y1 y2;zeros(1,1000) ones(1,1000)]; % Create training data
rng(22)
dat = dat(:,randperm(length(dat))); % Shuffle training data
%%
net8 = patternnet(20);
net8.trainParam.lr = 0.05;
net8.trainFcn = 'traingd';
net8.performFcn = 'mse';
%net8.divideFcn = 'dividetrain';
net8 = configure(net8,[dat(1,:); dat(2,:)],dat(3,:));
net8.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net8 = init(net8);
net8.iw{1,1}=iw_1;
net8.lw{2,1}=lw_1';
net8.b{1}= zeros(numofneurons,1);
[net8, tr8] = train(net8,[dat(1,:); dat(2,:)],dat(3,:));
yout8 = sim(net8,[x3 x4;y3 y5])>0.5;
%%
net9 = patternnet(20);
net9.trainParam.lr = 0.05;
net9.trainFcn = 'traingdm';
net9.performFcn = 'mse';
%net9.divideFcn = 'dividetrain';
net9 = configure(net9,[dat(1,:); dat(2,:)],dat(3,:));
net9.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net9 = init(net9);
net9.iw{1,1}=iw_1;
net9.lw{2,1}=lw_1';
net9.b{1}= zeros(numofneurons,1);
[net9, tr9] = train(net9,[dat(1,:); dat(2,:)],dat(3,:));
yout9 = sim(net9,[x3 x4;y3 y5])>0.5;
%%
net10 = patternnet(20);
net10.trainParam.lr = 0.05;
net10.trainFcn = 'trainlm';
net10.performFcn = 'mse';
%net10.divideFcn = 'dividetrain';
net10 = configure(net10,[dat(1,:); dat(2,:)],dat(3,:));
net10.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net10 = init(net10);
net10.iw{1,1}=iw_1;
net10.lw{2,1}=lw_1';
net10.b{1}= zeros(numofneurons,1);
[net10, tr10] = train(net10,[dat(1,:); dat(2,:)],dat(3,:));
yout10 = sim(net10,[x3 x4;y3 y5])>0.5;
%%
net11 = patternnet(20);
net11.trainParam.lr = 0.5;
net11.trainFcn = 'traingdm';
net11.performFcn = 'mse';
%net11.divideFcn = 'dividetrain';
net11 = configure(net11,[dat(1,:); dat(2,:)],dat(3,:));
net11.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net11 = init(net11);
net11.iw{1,1}=iw_1;
net11.lw{2,1}=lw_1';
net11.b{1}= zeros(numofneurons,1);
[net11, tr11] = train(net11,[dat(1,:); dat(2,:)],dat(3,:));
yout11 = sim(net11,[x3 x4;y3 y5])>0.5;
%%
net12 = patternnet(20);
net12.trainParam.lr = 0.005;
net12.trainFcn = 'traingdm';
net12.performFcn = 'mse';
%net12.divideFcn = 'dividetrain';
net12 = configure(net12,[dat(1,:); dat(2,:)],dat(3,:));
net12.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net12 = init(net12);
net12.iw{1,1}=iw_1;
net12.lw{2,1}=lw_1';
net12.b{1}= zeros(numofneurons,1);
[net12, tr12] = train(net12,[dat(1,:); dat(2,:)],dat(3,:));
yout12 = sim(net12,[x3 x4;y3 y5])>0.5;
%%
d=2;
y2 = radius2.*sin(angle2) - d;
y6 = radius4.*sin(angle4) - d;
dat = [x1 x2; y1 y2;zeros(1,1000) ones(1,1000)]; % Create training data
rng(22)
dat = dat(:,randperm(length(dat))); % Shuffle training data
%%
net13 = patternnet(20);
net13.trainParam.lr = 0.05;
net13.trainFcn = 'traingd';
net13.performFcn = 'mse';
%net13.divideFcn = 'dividetrain';
net13 = configure(net12,[dat(1,:); dat(2,:)],dat(3,:));
net13.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net13 = init(net13);
net13.iw{1,1}=iw_1;
net13.lw{2,1}=lw_1';
net13.b{1}= zeros(numofneurons,1);
[net13, tr13] = train(net13,[dat(1,:); dat(2,:)],dat(3,:));
yout13 = sim(net13,[x3 x4;y3 y6])>0.5;
%%
net14 = patternnet(20);
net14.trainParam.lr = 0.05;
net14.trainFcn = 'traingdm';
net14.performFcn = 'mse';
%net14.divideFcn = 'dividetrain';
net14 = configure(net14,[dat(1,:); dat(2,:)],dat(3,:));
net14.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net14 = init(net14);
net14.iw{1,1}=iw_1;
net14.lw{2,1}=lw_1';
net14.b{1}= zeros(numofneurons,1);
[net14, tr14] = train(net12,[dat(1,:); dat(2,:)],dat(3,:));
yout14 = sim(net14,[x3 x4;y3 y6])>0.5;
%%
net15 = patternnet(20);
net15.trainParam.lr = 0.05;
net15.trainFcn = 'trainlm';
net15.performFcn = 'mse';
%net15.divideFcn = 'dividetrain';
net15 = configure(net15,[dat(1,:); dat(2,:)],dat(3,:));
net15.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net15 = init(net15);
net15.iw{1,1}=iw_1;
net15.lw{2,1}=lw_1';
net15.b{1}= zeros(numofneurons,1);
[net15, tr15] = train(net15,[dat(1,:); dat(2,:)],dat(3,:));
yout15 = sim(net15,[x3 x4;y3 y6])>0.5;
%%
net16 = patternnet(20);
net16.trainParam.lr = 0.5;
net16.trainFcn = 'traingdm';
net16.performFcn = 'mse';
%net16.divideFcn = 'dividetrain';
net16 = configure(net16,[dat(1,:); dat(2,:)],dat(3,:));
net16.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net16 = init(net16);
net16.iw{1,1}=iw_1;
net16.lw{2,1}=lw_1';
net16.b{1}= zeros(numofneurons,1);
[net16, tr16] = train(net16,[dat(1,:); dat(2,:)],dat(3,:));
yout16 = sim(net16,[x3 x4;y3 y6])>0.5;
%%
net17 = patternnet(20);
net17.trainParam.lr = 0.005;
net17.trainFcn = 'traingdm';
net17.performFcn = 'mse';
%net17.divideFcn = 'dividetrain';
net17 = configure(net17,[dat(1,:); dat(2,:)],dat(3,:));
net17.trainParam.epochs = 5000;
numofneurons= 20;
rangeIW = sqrt(6)/(sqrt(numofneurons+2));
rng(51654);
in_w = 2*rangeIW.*rand(numofneurons*2,1)-rangeIW;
iw_1 = [in_w(1:numofneurons) in_w((numofneurons+1):(numofneurons*2))];
rangeLW = sqrt(6)/(sqrt(numofneurons+1));
lw_1 = 2*rangeLW.*rand(numofneurons,1)-rangeLW;
net17 = init(net17);
net17.iw{1,1}=iw_1;
net17.lw{2,1}=lw_1';
net17.b{1}= zeros(numofneurons,1);
[net17, tr17] = train(net17,[dat(1,:); dat(2,:)],dat(3,:));
yout17 = sim(net17,[x3 x4;y3 y6])>0.5;
%%
figure(1)
plotperform(tr1) % d=-8, backprop, 0.05, neuron = 20
%%
figure(2)
plotperform(tr2) % d=-8, backprop w/ mom, 0.05, neuron = 20
%%
figure(3)
plotperform(tr3) % d=-8, lm, 0.05, neuron = 20
%%
figure(4)
plotperform(tr4) % d=-8, backprop w/ mom, 0.5, neuron = 20
%%
figure(5)
plotperform(tr5) % d=-8, backprop w/ mom, 0.005, neuron = 20
%%
figure(6)
plotperform(tr6) % d=-8, backprop w/ mom, 0.05, neuron = 5
%%
figure(7)
plotperform(tr7) % d=-8, backprop w/ mom, 0.05, neuron = 40
%%
figure(8)
plotperform(tr8) % d=-4, backprop, 0.05, neuron = 20
%%
figure(9)
plotperform(tr9) % d=-4, backprop w/ mom, 0.05, neuron = 20
%%
figure(10)
plotperform(tr10) % d=-4, lm, 0.05, neuron = 20
%%
figure(11)
plotperform(tr11) % d=-4, backprop w/ mom, 0.5, neuron = 20
%%
figure(12)
plotperform(tr12) % d=-4, backprop w/ mom, 0.005, neuron = 20
%%
figure(13)
plotperform(tr13) % d=2, backprop, 0.05, neuron = 20
%%
figure(14)
plotperform(tr14) % d=2, backprop w/ mom, 0.05, neuron = 20
%%
figure(15)
plotperform(tr15) % d=2, lm, 0.05, neuron = 20
%%
figure(16)
plotperform(tr16) % d=2, backprop w/ mom, 0.5, neuron = 20
%%
figure(17)
plotperform(tr17) % d=2, backprop w/ mom, 0.005, neuron = 20
%%
figure(18)
hold on
plotconfusion(yout1,ydes) % d=-8, backprop, 0.05, neuron = 20
title('Confusion Matrix for d=-8, BP, 0.05, neuron = 20')
hold off
%%
figure(19)
hold on
plotconfusion(yout2,ydes) % d=-8, backprop w/ mom, 0.05, neuron = 20
title('Confusion Matrix for d=-8, backprop w/ mom, 0.05, neuron = 20')
hold off
%%
figure(20)
hold on
plotconfusion(yout3,ydes) % d=-8, lm, 0.05, neuron = 20
title('Confusion Matrix for d=-8, lm, 0.05, neuron = 20')
hold off
%%
figure(21)
hold on
plotconfusion(yout4,ydes) % d=-8, backprop w/ mom, 0.5, neuron = 20
title('Confusion Matrix for d=-8, backprop w/ mom, 0.5, neuron = 20')
hold off
%%
figure(22)
hold on
plotconfusion(yout5,ydes) % d=-8, backprop w/ mom, 0.005, neuron = 20
title('Confusion Matrix for d=-8, backprop w/ mom, 0.005, neuron = 20')
hold off
%%
figure(23)
hold on
plotconfusion(yout6,ydes) % d=-8, backprop w/ mom, 0.05, neuron = 5
title('Confusion Matrix for d=-8, backprop w/ mom, 0.05, neuron = 5')
hold off
%%
figure(24)
hold on
plotconfusion(yout7,ydes) % d=-8, backprop w/ mom, 0.05, neuron = 40
title('Confusion Matrix for d=-8, backprop w/ mom, 0.05, neuron = 40')
hold off
%%
figure(25)
hold on
plotconfusion(yout8,ydes) % d=-4, backprop, 0.05, neuron = 20
title('Confusion Matrix for d=-4, backprop, 0.05, neuron = 20')
hold off
%%
figure(26)
hold on
plotconfusion(yout9,ydes) % d=-4, backprop w/ mom, 0.05, neuron = 20
title('Confusion Matrix for d=-4, backprop w/ mom, 0.05, neuron = 20')
hold off
%%
figure(27)
hold on
plotconfusion(yout10,ydes) % d=-4, lm, 0.05, neuron = 20
title('Confusion Matrix for d=-4, lm, 0.05, neuron = 20')
hold off
%%
figure(28)
hold on
plotconfusion(yout11,ydes) % d=-4, backprop w/ mom, 0.5, neuron = 20
title('Confusion Matrix for d=-4, backprop w/ mom, 0.5, neuron = 20')
hold off
%%
figure(29)
hold on
plotconfusion(yout12,ydes) % d=-4, backprop w/ mom, 0.005, neuron = 20
title('Confusion Matrix for d=-4, backprop w/ mom, 0.005, neuron = 20')
hold off
%%
figure(30)
hold on
plotconfusion(yout13,ydes) % d=2, backprop, 0.05, neuron = 20
title('Confusion Matrix for d=2, backprop, 0.05, neuron = 20')
hold off
%%
figure(31)
hold on
plotconfusion(yout14,ydes) % d=2, backprop w/ mom, 0.05, neuron = 20
title('Confusion Matrix for d=2, backprop w/ mom, 0.05, neuron = 20')
hold off
%%
figure(32)
hold on
plotconfusion(yout15,ydes) % d=2, lm, 0.05, neuron = 20
title('Confusion Matrix for d=2, lm, 0.05, neuron = 20')
hold off
%%
figure(33)
hold on
plotconfusion(yout16,ydes) % d=2, backprop w/ mom, 0.5, neuron = 20
title('Confusion Matrix for d=2, backprop w/ mom, 0.5, neuron = 20')
hold off
%%
figure(34)
hold on
plotconfusion(yout17,ydes) % d=2, backprop w/ mom, 0.005, neuron = 20
title('Confusion Matrix for d=2, backprop w/ mom, 0.005, neuron = 20')
hold off
%%
figure(35)
plot(x3,y3,'+',x4,y4,'r+')
hold on
for ii = 1:P
z = sim(net1,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-8, backprop, 0.05, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(36)
plot(x3,y3,'+',x4,y4,'r+')
hold on
for ii = 1:P
z = sim(net2,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-8, backprop w/ mom, 0.05, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(37)
plot(x3,y3,'+',x4,y4,'r+')
hold on
for ii = 1:P
z = sim(net3,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-8, lm, 0.05, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(38)
plot(x3,y3,'+',x4,y4,'r+')
hold on
for ii = 1:P
z = sim(net4,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-8, backprop w/ mom, 0.5, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(39)
plot(x3,y3,'+',x4,y4,'r+')
hold on
for ii = 1:P
z = sim(net5,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-8, backprop w/ mom, 0.005, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(40)
plot(x3,y3,'+',x4,y4,'r+')
hold on
for ii = 1:P
z = sim(net6,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-8, backprop w/ mom, 0.05, neuron = 5')
hold off
legend('Class 0','Class 1')
%%
figure(41)
plot(x3,y3,'+',x4,y4,'r+')
hold on
for ii = 1:P
z = sim(net7,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-8, backprop w/ mom, 0.05, neuron = 40')
hold off
legend('Class 0','Class 1')
%%
figure(42)
plot(x3,y3,'+',x4,y5,'r+')
hold on
for ii = 1:P
z = sim(net8,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-4, backprop, 0.05, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(43)
plot(x3,y3,'+',x4,y5,'r+')
hold on
for ii = 1:P
z = sim(net9,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-4, backprop w/ mom, 0.05, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(44)
plot(x3,y3,'+',x4,y5,'r+')
hold on
for ii = 1:P
z = sim(net10,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-4, lm, 0.05, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(45)
plot(x3,y3,'+',x4,y5,'r+')
hold on
for ii = 1:P
z = sim(net11,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-4, backprop w/ mom, 0.5, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(46)
plot(x3,y3,'+',x4,y5,'r+')
hold on
for ii = 1:P
z = sim(net12,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=-4, backprop w/ mom, 0.005, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(47)
plot(x3,y3,'+',x4,y6,'r+')
hold on
for ii = 1:P
z = sim(net13,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=2, backprop, 0.05, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(48)
plot(x3,y3,'+',x4,y6,'r+')
hold on
for ii = 1:P
z = sim(net14,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=2, backprop w/ mom, 0.05, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(49)
plot(x3,y3,'+',x4,y6,'r+')
hold on
for ii = 1:P
z = sim(net15,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=2, lm, 0.05, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(50)
plot(x3,y3,'+',x4,y6,'r+')
hold on
for ii = 1:P
z = sim(net16,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=2, backprop w/ mom, 0.5, neuron = 20')
hold off
legend('Class 0','Class 1')
%%
figure(51)
plot(x3,y3,'+',x4,y6,'r+')
hold on
for ii = 1:P
z = sim(net17,[x(:,ii)'; y(:,ii)']);
z = logical(z>0.5);
z = logical(diff(z));
b = find(z==1);
plot(x(b,ii),y(b,ii),'k.','LineWidth',6)
end
title('d=2, backprop w/ mom, 0.005, neuron = 20')
hold off
legend('Class 0','Class 1')