import Augmentor 
p=Augmentor.Pipeline("/home/xjw/test")
p.rotate(probability=0.5,max_left_rotation=25,max_right_rotation=10)
p.random_distortion(probability=1,grid_height=5,grid_width=16,magnitude=8)
p.shear(probability=1,max_shear_left=15,max_shear_right=15)
p.skew(probability=1)
p.flip_top_bottom(probability=1)
p.sample(10)