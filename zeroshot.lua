require 'nn'
require 'optim'
require 'HingeCriterion'
require 'xlua'
matio = require 'matio'
Y = matio.load('data.mat','Y')
X = matio.load('data.mat','X')
lambda = matio.load('data.mat','lambda')
V_gtruth = matio.load('data.mat','V_gtruth')
data = matio.load('data.mat')

trainLogger = optim.Logger(paths.concat('/home/dhill/abd/logs/','train_zeroshot.log'))
----------------------------------------------------------------------------------------
function splitDataset(d, l, Y,ratio1, ratio2)
   --local shuffle = torch.randperm(d:size(1))
   local numTrain = math.floor(d:size(1) * ratio1)
   local numVal = math.floor(d:size(1)*ratio2)
   local numTest = d:size(1) - numTrain - numVal
   local trainData = torch.Tensor(numTrain, d:size(2))
   local ValData = torch.Tensor(numVal,d:size(2))
   local testData = torch.Tensor(numTest, d:size(2))
   local trainLabels = torch.Tensor(numTrain,l:size(2))
   local ValLabels = torch.Tensor(numVal,l:size(2))
   local testLabels = torch.Tensor(numTest,l:size(2))
   local Y_traindata = torch.Tensor(numTrain,Y:size(2))
   local Y_Valdata = torch.Tensor(numVal,Y:size(2))
   local Y_testdata = torch.Tensor(numTest,Y:size(2))
   for i=1,numTrain do
      trainData[i] = d[i]:clone()
      trainLabels[i] = l[i]:clone()
      Y_traindata[i] = Y[i]:clone()
   end
   for i=numTrain+1,numVal do
	  ValData[i] = d[i]:clone()
	  ValLabels[i] = l[i]:clone()
          Y_Valdata[i] = Y[i]:clone()
   end
   for i=numTrain+numVal+1,numTrain+numVal+numTest do
      testData[i-numTrain-numVal] = d[i]:clone()
      testLabels[i-numTrain-numVal] = l[i]:clone()
      Y_testdata[i-numTrain-numVal] = Y[i]:clone()
   end
   return trainData, trainLabels, ValData,ValLabels,testData, testLabels,Y_traindata,Y_Valdata,Y_testdata
 
end
trainData, trainLabels,testData, testLabels,Y_traindata,Y_testdata = splitDataset(X,lambda,Y, 0.6,0.2) -- 60/20/20 split
--------------------------------------------------------------------------------------------------------------------------------------
Vo = torch.rand(2,3)
Vo = Vo:transpose(1,2)
epoch = epoch or 1
MAXepoch = 1.5e2+1

train_data = trainData*Vo
val_data = ValData*V_gtruth
test_data = testData*V_gtruth

batchsize = trainData:size(1)
----frobenius norm of Vo----------------
frob_norm = torch.abs(Vo):pow(2):sum()
----------------------------------------------
-------model-----------------------------
module = nn.Linear(3,2)
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
inputs_x = torch.Tensor(Xr:size(1),3):double() 
inputs_l = torch.Tensor(lr:size(1),2):double()
inputs_y = torch.Tensor(Yr:size(1),1):double()
local targets ={}
for i = 1,Xr:size(1) do
  
  xlua.progress(i,Xr:size(1))
  --for j = i, math.min(i+batchsize-1,Xr:size(1)) do
        inputs_x[{m,{1,3}}]:copy(Xr[shuffle[i]])
        inputs_l[{m,{1,2}}]:copy(lr[shuffle[i]])
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
                  --print(dl_dv3:size())
                  --print(module.weight,module.gradWeight)
                  model:backward(inputs_x,dl_dv3)
                  trainloss = trainloss+l
                 return l,gradParameters
               end
  
  optim.sgd(feval,parameters,sgdstate)
  trainloss = trainloss/Xr:size(1)
  epoch = epoch + 1
  return trainloss
end
                 
-------------------------Validation----------------------------------------
-----------------------------------Testing-------------------------------------------------
function test(Xt,Lt,Yt)
model:evaluate()

local testErr = 0
  local inputs = {}
  inputs_x = torch.Tensor(Xt:size(1),3):double() 
  inputs_l = torch.Tensor(Lt:size(1),2):double()
  inputs_y = torch.Tensor(Yt:size(1),1):double()
  local targets ={}
        inputs_x:copy(Xt)
        inputs_l:copy(Lt)
        inputs_y:copy(Yt)
        
  local output = model:forward(inputs_x)
  output = (output*inputs_l:transpose(1,2)):diag()
  output = output:sign()
  output = torch.eq(output,inputs_y)
  output = torch.sum(output)
  mean = output/Xt:size(1)
  print ('Classification accuracy is:'..mean)

end

while (epoch<MAXepoch) do
  trainloss = train(trainData,trainLabels,Y_traindata)
  --valerr = evaluate(ValData,ValLabels,Y_Valdata)
  trainLogger:add{['% train loss'] = trainloss}--,    ['% Val loss'] = valerr}
  --trainLogger:style{['% train loss'] = '-'}--  ['% Val error']= '-'}
  --trainLogger:plot()
end
test(testData,testLabels,Y_testdata)




