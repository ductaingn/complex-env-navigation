import raisimpy

world = raisimpy.World()
anymal_ = world.addArticulatedSystem('rsc/husky/husky.urdf')
# anymal_ = world.addArticulatedSystem('rsc/anymal_c/urdf/anymal.urdf')
gcDim_ = anymal_.getGeneralizedCoordinateDim()
gvDim_ = anymal_.getDOF()                
nJoints_ = gvDim_ - 6  
print(gcDim_)
print(gvDim_)
