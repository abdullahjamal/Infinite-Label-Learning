local HingeCriterion,parent = torch.class('nn.HingeCriterion','nn.Criterion')
 
function HingeCriterion:_init(margin)
    parent:_init(self)
    margin = margin or 1
    self.margin = margin
end

function HingeCriterion:updateOutput(input,target)    
    local op = ((input:neg()):cmul(target)):add(1) 
    self.output= ((op:cmax(0)):sum())/input:size(1)
    return self.output,op
end

function HingeCriterion:updateGradInput(input,gradOutput)
     
    local pwm = torch.gt(input[3],0):double()
    gradOutput = pwm:cmul(gradOutput:neg())
    input[2][{{1,input[2]:size(1)}, 1}]:cmul(gradOutput)
    input[2][{{1,input[2]:size(1)}, 2}]:cmul(gradOutput)
    self.gradInput= (input[2])		
    
  return self.gradInput 
end
