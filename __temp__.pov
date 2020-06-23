#include "colors.inc"
light_source {
<2,4,-3>
color
<1,1,1> 
}
sphere {
<0,1,2>
2
texture {
pigment {
color
<1,0,1> 
} 
} 
}
camera {
location
<0,2,-3>
look_at
<0,1,2>
right
<1.8,0,0> 
}
global_settings{

}