require 'nn'
require 'optim'
require 'HingeCriterion'
require 'xlua'
matio = require 'matio'
--Y = matio.load('data_22.mat','Y')
--X = matio.load('data_22.mat','X')
--lambda = matio.load('data_22.mat','lambda')
--V_gtruth = matio.load('data_22.mat','V')
Y2 = matio.load('data_new2.mat','Y1')
Y1 = matio.load('data_new2.mat','Y')
X1 = matio.load('data_new2.mat','X')
lambda1 = matio.load('data_new2.mat','lambda1')
V1= matio.load('data_new2.mat','V')

traindata_1 = torch.Tensor(500,10):copy(X1[{{1,500},{1,10}}])
--traindata_2 = torch.Tensor(1000,10):copy(X1[{{501,1500},{1,10}}])
--traindata_3 = torch.Tensor(2000,10):copy(X1[{{1501,3500},{1,10}}])
--traindata_4 = torch.Tensor(4000,10):copy(X1[{{12001,16000},{1,10}}])

trainlabel_1 = torch.Tensor(500,9):copy(lambda1[{{1,500},{1,9}}])
trainlabel_2 = torch.Tensor(1000,9):copy(lambda1[{{501,1500},{1,9}}])
trainlabel_3 = torch.Tensor(2000,9):copy(lambda1[{{1501,3500},{1,9}}])
trainlabel_4 = torch.Tensor(4000,9):copy(lambda1[{{12001,16000},{1,9}}])

--Y_traindata1 = torch.Tensor(100,500):copy(Y2[{{1,100},{1,500}}])
--Y_traindata2 = torch.Tensor(100,1000):copy(Y2[{{1,100},{501,1500}}])
--Y_traindata3 = torch.Tensor(100,2000):copy(Y2[{{1,100},{1501,3500}}])
--Y_traindata4 = torch.Tensor(100,4000):copy(Y2[{{1,100},{12001,16000}}])

Y_traindata1 = ((traindata_1*V1)*(trainlabel_1:transpose(1,2))):sign()
Y_traindata2 = ((traindata_1*V1)*(trainlabel_2:transpose(1,2))):sign()
Y_traindata3 = ((traindata_1*V1)*(trainlabel_3:transpose(1,2))):sign()
Y_traindata4 = ((traindata_1*V1)*(trainlabel_4:transpose(1,2))):sign()


--Y_traindata5 = torch.Tensor(500,100):copy(Y2[{{1,500},{1,100}}])
--Y_traindata6 = torch.Tensor(1000,100):copy(Y2[{{501,1500},{1,100}}])
--Y_traindata7 = torch.Tensor(2000,100):copy(Y2[{{1501,3500},{1,100}}])
--Y_traindata8 = torch.Tensor(4000,100):copy(Y2[{{12001,16000},{1,100}}])

--test_1000 = torch.Tensor(1000,10):copy(X1[{{1001,2000},{1,10}}])
--test_2000 = torch.Tensor(2000,10):copy(X1[{{2001,4000},{1,10}}])
--test_3000 = torch.Tensor(3000,10):copy(X1[{{4001,7000},{1,10}}])
test_5000 = torch.Tensor(5000,10):copy(X1[{{5001,10000},{1,10}}])
--test_10000 = torch.Tensor(10000,10):copy(X1[{{12001,22000},{1,10}}])

--testlabels_1000 = torch.Tensor(1000,9):copy(lambda1[{{1001,2000},{1,9}}])
--testlabels_2000 = torch.Tensor(2000,9):copy(lambda1[{{2001,4000},{1,9}}])
--testlabels_3000 = torch.Tensor(3000,9):copy(lambda1[{{4001,7000},{1,9}}])
testlabels_5000 = torch.Tensor(5000,9):copy(lambda1[{{5001,10000},{1,9}}])
--testlabels_10000 = torch.Tensor(10000,9):copy(lambda1[{{12001,22000},{1,9}}])

Yt_5000 = ((test_5000*V1)*(testlabels_5000:transpose(1,2))):sign()
--Y_1000 = torch.Tensor(1000,1):copy(Y1[{{1001,2000},{1}}])
--Y_2000 = torch.Tensor(2000,1):copy(Y1[{{2001,4000},{1}}])
--Y_3000 = torch.Tensor(3000,1):copy(Y1[{{4001,7000},{1}}])
Y_5000 = torch.Tensor(5000,1):copy(Y1[{{5001,10000},{1}}])
--Y_10000 = torch.Tensor(10000,1):copy(Y1[{{12001,22000},{1}}])

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
   local Y_trainData = torch.Tensor(numTrain,Y:size(2))
   local Y_testdata = torch.Tensor(numTest,Y:size(2))
   for i=1,numTrain do
      trainData[i] = d[i]:clone()
      trainLabels[i] = l[i]:clone()
      Y_trainData[i] = Y[i]:clone()
   end
   for i=numTrain+1,numTrain+numTest do
      testData[i-numTrain] = d[i]:clone()
      testLabels[i-numTrain] = l[i]:clone()
      Y_testdata[i-numTrain] = Y[i]:clone()
   end
   return trainData, trainLabels,testData, testLabels,Y_trainData,Y_testdata
 
end
--trainData, trainLabels,testData,testLabels,Y_trainData,Y_testdata = splitDataset(X1,lambda1,Y1, 0.5) -- 50/50 split
--------------------------------------------------------------------------------------------------------------------------------------
Vo = torch.rand(9,10)
Vo = Vo:transpose(1,2)
epoch = epoch or 1
MAXepoch = 15e2+1

batchsize = traindata_1:size(1)
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
   learningRate = 0.3, 
   learningRateDecay =0,
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
local  h_error  = 0
shuffle = torch.randperm(lr:size(1))
print('trainer on training set:')
print("training: online epoch#" .. epoch.. '[batchSize=' ..batchsize .. ']')
local inputs = {}
local m = 1
inputs_x = torch.Tensor(Xr:size(1),10):double() 
inputs_l = torch.Tensor(lr:size(1),9):double()
inputs_y = torch.Tensor(Xr:size(1),lr:size(1)):double()
local targets ={}
for i = 1,lr:size(1) do
  
  xlua.progress(i,Xr:size(1))
  --for j = i, math.min(i+batchsize-1,Xr:size(1)) do
    --    inputs_x[{m,{1,10}}]:copy(Xr[shuffle[i]])
      inputs_l[{m,{1,9}}]:copy(lr[shuffle[i]])
  --      inputs_y[{m,{1,1}}]:copy(Yr[shuffle[i]])
        m = m+1
end
inputs_x:copy(Xr)
        --inputs_l:copy(lr)
        inputs_y:copy(Yr)
table.insert(inputs,inputs_x)
table.insert(inputs,inputs_l)
  local feval = function(x)
                 if x~=parameters then
                    parameters:copy(x)
                 end
                 local l = 0
                 local output = model:forward(inputs_x)
                 output = output*(inputs_l:transpose(1,2))
                 --output = output:diag()
                  --print(inputs_y)
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
  trainloss = trainloss/Xr:size(1)
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
  inputs_y = torch.Tensor(5000,1):double()
  local targets ={}
        inputs_x:copy(Xt)
        inputs_l:copy(Lt)
        inputs_y:copy(Yt)
        
  local output = model:forward(inputs_x)
  output = (output*inputs_l:transpose(1,2)):diag()
  output = output:sign()
  output1 = torch.ne(output,inputs_y)
  --print(output1:size())
  output1 = torch.sum(output1)
  print(output1)
  h_error = output1/(Xt:size(1))
  return h_error
end

while (epoch<MAXepoch) do
  trainloss = train(traindata_1,trainlabel_2,Y_traindata2)
end
err1 = test(test_5000,testlabels_5000,Y_5000)
------------------------------------------------
--epoch = 1
--batchsize = traindata_2:size(1)
--model:parameters()[1]:copy(Vo)
--parameters,gradParameters = model:getParameters()
--while (epoch<MAXepoch) do
  --trainloss = train1(traindata_1,trainlabel_2,Y_traindata2)
--end
--err2 = test(test_5000,testlabels_5000,Y_5000)
-------------------------------------------------
--epoch = 1
--batchsize = traindata_3:size(1)
--model:parameters()[1]:copy(Vo)
--parameters,gradParameters = model:getParameters()
--while (epoch<MAXepoch) do
  --trainloss = train2(traindata_1,trainlabel_3,Y_traindata3)
--end
--err3 = test(test_5000,testlabels_5000,Y_5000)
-----------------------------------------------------
--epoch = 1
--batchsize = traindata_4:size(1)
--model:parameters()[1]:copy(Vo)
--parameters,gradParameters = model:getParameters()
--while (epoch<MAXepoch) do
  --trainloss = train3(traindata_1,trainlabel_4,Y_traindata4)
--end
--err4 = test(test_5000,testlabels_5000,Y_5000)
-----------------------------------------------------

print('Error for 500 labels:' ..err1)

--print('Error for 1000 labels:'..err2)

--print('Error for 2000 labels:'..err3)

--print('Error for 4000 labels:'..err4)



