import raisimpy

world = raisimpy.World()
world.exportToXml('test.xml')
anymal_ = world.addArticulatedSystem('rsc/husky/husky.urdf')
