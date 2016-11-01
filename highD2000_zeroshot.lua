require 'nn'
require 'optim'
require 'HingeCriterion'
require 'xlua'
matio = require 'matio'
Y4 = matio.load('/home/dhill/abd/data/data_5.mat','Y')
X4 = matio.load('/home/dhill/abd/data/data_5.mat','X')
lambda4 = matio.load('/home/dhill/abd/data/data_5.mat','lambda1')
V_gtruth4 = matio.load('/home/dhill/abd/data/data_5.mat','V')

Y3 = matio.load('/home/dhill/abd/data/data_4.mat','Y')
X3 = matio.load('/home/dhill/abd/data/data_4.mat','X')
lambda3 = matio.load('/home/dhill/abd/data/data_4.mat','lambda1')
V_gtruth3 = matio.load('/home/dhill/abd/data/data_4.mat','V')

Y2 = matio.load('/home/dhill/abd/data/data_3.mat','Y')
X2 = matio.load('/home/dhill/abd/data/data_3.mat','X')
lambda2 = matio.load('/home/dhill/abd/data/data_3.mat','lambda1')
V_gtruth2 = matio.load('/home/dhill/abd/data/data_3.mat','V')

Y1 = matio.load('/home/dhill/abd/data/data_2.mat','Y')
X1 = matio.load('/home/dhill/abd/data/data_2.mat','X')
lambda1 = matio.load('/home/dhill/abd/data/data_2.mat','lambda1')
V_gtruth1 = matio.load('/home/dhill/abd/data/data_2.mat','V')

Y = matio.load('/home/dhill/abd/data/data_1.mat','Y')
X = matio.load('/home/dhill/abd/data/data_1.mat','X')
lambda = matio.load('/home/dhill/abd/data/data_1.mat','lambda1')
V_gtruth = matio.load('/home/dhill/abd/data/data_1.mat','V')

trainLogger = optim.Logger(paths.concat('/home/dhill/abd/logs/','train_zeroshot_2000.log'))
----------------------------------------------------------------------------------------
function splitDataset(d, l, Y,ratio1)
   --local shuffle = torch.randperm(d:size(1))
   local numTrain = math.floor(d:size(1) * ratio1)
   local numTest = d:size(1) - numTrain
   local trainData = torch.Tensor(numTrain, d:size(2))
   local testData = torch.Tensor(numTest, d:size(2))
   local trainLabels = torch.Tensor(numTrain,l:size(2))
   local testLabels = torch.Tensor(numTest,l:size(2))
   local Y_traindata = torch.Tensor(numTrain,Y:size(2))
   local Y_testdata = torch.Tensor(numTest,Y:size(2))
   for i=1,numTrain do
      trainData[i] = d[i]:clone()
      trainLabels[i] = l[i]:clone()
      Y_traindata[i] = Y[i]:clone()
   end
   for i=numTrain+1,numTrain+numTest do
      testData[i-numTrain] = d[i]:clone()
      testLabels[i-numTrain] = l[i]:clone()
      Y_testdata[i-numTrain] = Y[i]:clone()
   end
   return trainData, trainLabels,testData, testLabels,Y_traindata,Y_testdata
 
end
trainData, trainLabels,testData,testLabels,Y_traindata,Y_testdata = splitDataset(X,lambda,Y, 0.5) -- 50/50 split
trainData_1, trainLabels_1,testData_1,testLabels_1,Y_traindata_1,Y_testdata_1 = splitDataset(X1,lambda1,Y1, 0.5) -- 50/50 split
trainData_2, trainLabels_2,testData_2,testLabels_2,Y_traindata_2,Y_testdata_2 = splitDataset(X2,lambda2,Y2, 0.5) -- 50/50 split
trainData_3, trainLabels_3,testData_3,testLabels_3,Y_traindata_3,Y_testdata_3 = splitDataset(X3,lambda3,Y3, 0.5) -- 50/50 split
trainData_4, trainLabels_4,testData_4,testLabels_4,Y_traindata_4,Y_testdata_4 = splitDataset(X4,lambda4,Y4, 0.5) -- 50/50 split
--------------------------------------------------------------------------------------------------------------------------------------
ValData = torch.randn(1000,100)
Vo = torch.rand(9,10)
Vo = Vo:transpose(1,2)
epoch = epoch or 1
MAXepoch = 4e2+1

batchsize = trainData:size(1)
----frobenius norm of Vo----------------
--frob_norm = torch.abs(Vo):pow(2):sum()
----------------------------------------------
-------model-----------------------------
module = nn.Linear(10,9)
model= nn.Sequential()
model:add(module)
model:parameters()[1]:copy(Vo)
print(model)
parameters,gradParameters = model:getParameters()

sgdstate = {
   learningRate = 1e-3, 
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}
---------------------------------------------------
------------Loss----------------------

crit = nn.HingeCriterion()

--------------Training-----------------------------------------
function train(Xr,lr,Yr)
model:training()

local trainloss = 0

shuffle = torch.randperm(Xr:size(1))
print('trainer on training set:')
print("training: online epoch#" .. epoch.. '[batchSize=' ..batchsize .. ']')
local inputs = {}
local m = 1
inputs_x = torch.Tensor(Xr:size(1),10):double() 
inputs_l = torch.Tensor(lr:size(1),9):double()
inputs_y = torch.Tensor(Yr:size(1),1):double()
local targets ={}
for i = 1,Xr:size(1) do
  
  xlua.progress(i,Xr:size(1))
        inputs_x[{m,{1,10}}]:copy(Xr[shuffle[i]])
        inputs_l[{m,{1,9}}]:copy(lr[shuffle[i]])
        inputs_y[{m,{1,1}}]:copy(Yr[shuffle[i]])
        m = m+1
end
table.insert(inputs,inputs_x)
table.insert(inputs,inputs_l)
  local feval = function(x)
                 if x~=parameters then
                    parameters:copy(x)
                 end
                 local l = 0
                 local output = model:forward(inputs_x)
                 output = output*(inputs_l:transpose(1,2))
                 output = output:diag()
                  local loss1,out = crit:forward(output,inputs_y)
                  table.insert(inputs,out)
                  l = l + loss1
                  local dl_dv3 = crit:backward(inputs,inputs_y)
                  model:zeroGradParameters()
                  model:backward(inputs_x,dl_dv3)
                  trainloss = trainloss+l
                 return l,gradParameters
               end
  
  optim.sgd(feval,parameters,sgdstate)
  trainloss = trainloss
  epoch = epoch + 1
  return trainloss
end


-----------------------------------Testing-------------------------------------------------
function test(Xt,Lt,Yt)
model:evaluate()

local testErr = 0
  local inputs = {}
  inputs_x = torch.Tensor(Xt:size(1),10):double() 
  inputs_l = torch.Tensor(Lt:size(1),9):double()
  inputs_y = torch.Tensor(Yt:size(1),1):double()
  local targets ={}
        inputs_x:copy(Xt)
        inputs_l:copy(Lt)
        inputs_y:copy(Yt)
        
  local output = model:forward(inputs_x)
  output = (output*inputs_l:transpose(1,2)):diag()
  output = output:sign()
  --output = torch.eq(output,inputs_y)
  output1 = torch.ne(output,inputs_y)
  output1 = torch.sum(output1)
  h_error = output1/Xt:size(1)
  --mean = output/Xt:size(1)
  return h_error

end
-----------------------------2000---------------------------------------------
while (epoch<MAXepoch) do
  trainloss = train(trainData,trainLabels,Y_traindata)
  --trainLogger:add{['% train loss'] = trainloss --, ['% Val loss'] = valerr}
  --trainLogger:style{['% train loss'] = '-'--, ['% Val loss'] = '-'}
  --trainLogger:plot()
end
err = test(testData,testLabels,Y_testdata)
------------------------------4000------------------------------------------------
epoch = 1
batchsize = trainData_1:size(1)
model:parameters()[1]:copy(Vo)
parameters,gradParameters = model:getParameters()
while (epoch<MAXepoch) do
  trainloss1 = train(trainData_1,trainLabels_1,Y_traindata_1)
  --trainLogger:add{['% train loss1'] = trainloss}
  --trainLogger:style{['% train loss'] = '-'}
  --trainLogger:plot()
end
err1 = test(testData_1,testLabels_1,Y_testdata_1)
------------------------------6000--------------------------------------------------
epoch = 1
batchsize = trainData_2:size(1)
model:parameters()[1]:copy(Vo)
parameters,gradParameters = model:getParameters()
while (epoch<MAXepoch) do
  trainloss2 = train(trainData_2,trainLabels_2,Y_traindata_2)
  --trainLogger:add{['% train loss2'] = trainloss}
  --trainLogger:style{['% train loss'] = '-'}
  --trainLogger:plot()
end
err2 = test(testData_2,testLabels_2,Y_testdata_2)
-------------------------------10000-------------------------------------------------
epoch = 1
batchsize = trainData_3:size(1)
model:parameters()[1]:copy(Vo)
parameters,gradParameters = model:getParameters()
while (epoch<MAXepoch) do
  trainloss3 = train(trainData_3,trainLabels_3,Y_traindata_3)
  --trainLogger:add{['% train loss3'] = trainloss}
  --trainLogger:style{['% train loss'] = '-'}
  --trainLogger:plot()
end
err3= test(testData_3,testLabels_3,Y_testdata_3)
-------------------------------20000----------------------------------------------------
epoch = 1
batchsize = trainData_4:size(1)
model:parameters()[1]:copy(Vo)
parameters,gradParameters = model:getParameters()
while (epoch<MAXepoch) do
  trainloss4 = train(trainData_4,trainLabels_4,Y_traindata_4)
  --trainLogger:add{['% train loss4'] = trainloss}
  --trainLogger:style{['% train loss'] = '-'}
  --trainLogger:plot()
end
err4 = test(testData_4,testLabels_4,Y_testdata_4)
----------------------------------------------------------------------------------
print ('Error for 1000labels :'..err)
print ('Error for 2000labels :'..err1)
print ('Error for 3000labels :'..err2)
print ('Error for 5000labels :'..err3)
print ('Error for 10000labels :'..err4)


