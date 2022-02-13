import bpy
import mathutils
from mathutils import Vector
import bmesh
from bpy import context
import random
from mathutils import geometry
from math import sqrt

"""
Function to read control points from file
Checks
1. 4 sets of points provided
2. Each curve shares one endpoint
"""
def read_Control_Points(fileName):
    ctrl_pts = {}
    num_ctrl_pts = 4
    line_cnt = 0
    curve_count = 0
    vertex_count=0
    vertex_line_length = 4
    scale_factor = 1
    for line in open(fileName,"r"):
        #skip comments
        if line.startswith('#'):continue
        #first line - read number of curves specified. Assert that it is 4
        if line_cnt==0:
            value=int((line.split())[0])
            assert value == num_ctrl_pts
        else:    
            #next pattern should be number of control points followed by correct number of entries
            values=line.split()
            #skip empty lines
            if not values:continue
            #lines containing vertex values
            if values[0]=='v':
                #4 entries expected in a line that starts with v
                assert len(values) == vertex_line_length
                #check that number of vertices specified is within limit
                assert vertex_count < ctrl_pts[curve_count]['N']
                #add to dictionary
                vertex = []
                vertex.append(scale_factor*float(values[1]))
                vertex.append(scale_factor*float(values[2]))
                vertex.append(scale_factor*float(values[3]))
                #ctrl_pts[curve_count]['points'].append(tuple(vertex))
                ctrl_pts[curve_count]['points'].append(vertex)
                vertex_count = vertex_count + 1
            else:
                #this line is expected to contain number of points for next curve
                assert len(values) == 1
                assert int(values[0]) > 0
                #check that exact number of points have been read
                if curve_count > 0:
                    assert len(ctrl_pts[curve_count]['points']) == ctrl_pts[curve_count]['N']
                #initialise for next curve
                curve_count = curve_count + 1
                assert curve_count <= num_ctrl_pts
                ctrl_pts[curve_count] = {}
                ctrl_pts[curve_count]['points'] = []
                ctrl_pts[curve_count]['N'] = int(values[0])
                vertex_count = 0                   
        #update vars for next iteration
        line_cnt = line_cnt + 1
    #check exact number of points condition for last curve
    assert len(ctrl_pts[curve_count]['points']) == ctrl_pts[curve_count]['N']
    #check that each curve shares one endpoint
    for curve in ctrl_pts.keys():
        ctrl_pts1 = ctrl_pts[curve]['points']
        if curve == 1:
            ctrl_pts2 = ctrl_pts[num_ctrl_pts]['points']
        else:
            ctrl_pts2 = ctrl_pts[curve-1]['points']
        assert ctrl_pts1[0] == ctrl_pts2[-1]    
    #return value
    return ctrl_pts

"""
Common computations and rendering functions
"""
#get list of evenly spaced parameter values. Needed for evaluation of parametric curves
def get_t_values(lower,upper,num_divisions):
    #set t values at which we want to evaluate the curve
    lower = 0
    upper = 1
    t_values = [lower + x*(upper-lower)/num_divisions for x in range(num_divisions)]
    t_values.append(upper)
    return t_values

#split points as lists in x,y,z directions
def get_coord_list(points):
    coords = {}
    coords['x'] = []
    coords['y'] = []
    coords['z'] = []
    for pt in points:
        coords['x'].append(pt[0])
        coords['y'].append(pt[1])
        coords['z'].append(pt[2])
    return coords

#triangulate object
def triangulate_object(obj):
    me = obj.data
    # Get a BMesh representation
    bm = bmesh.new()
    bm.from_mesh(me)

    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()
    return me

#create mesh from data
def createMeshFromData(name, origin, verts, faces):
    # Create mesh and object
    me = bpy.data.meshes.new(name+'Mesh')
    ob = bpy.data.objects.new(name, me)
    ob.location = origin
    ob.show_name = True
 
    scn = bpy.context.scene
    scn.collection.objects.link(ob)
    
    bpy.context.view_layer.objects.active=ob
    ob.select_set(True)
     
    # Create mesh from given verts, faces.
    me.from_pydata(verts, [], faces)
    # Update mesh with new data
    me=triangulate_object(ob)
    me.update()    
    return ob

#export object to file
def export_obj(filepath,obj):
    mesh = obj.data
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in mesh.vertices:
            f.write("v %.4f %.4f %.4f\n" % v.co[:])
        for p in mesh.polygons:
            f.write("f")
            for i in p.vertices:
                f.write(" %d" % (i + 1))
            f.write("\n")

"""
Functions for generating bezier curves
"""
#de casteljau algorithm
def de_casteljau(t, coeffs, num_coeffs):
    beta = [coeff for coeff in coeffs]
    for j in range(1,num_coeffs):
        for k in range(num_coeffs-j):
            beta[k] = beta[k] * (1 - t) + beta[k + 1] * t
    return beta[0]

#compute bezier curve points        
def compute_bezier_curves(ctrl_pts, num_divisions):
    bezier_curves = {}
    #set t values at which we want to evaluate the curve
    t_values = get_t_values(0,1,num_divisions)
    #for each set of control points  
    for curve in ctrl_pts.keys():
        bezier_curves[curve] = []
        coord_list = get_coord_list(ctrl_pts[curve]['points'])
        num_coeffs = ctrl_pts[curve]['N']
        #run decasteljau on x,y,z separately
        for t in t_values:
            c_t = []
            c_t.append(de_casteljau(t, coord_list['x'], num_coeffs))
            c_t.append(de_casteljau(t, coord_list['y'], num_coeffs))
            c_t.append(de_casteljau(t, coord_list['z'], num_coeffs))
            bezier_curves[curve].append(c_t)
    #return values
    return bezier_curves                            

#make surface from pair of curves
def make_faces(verts, num_pts):
    faces = []
    lim = num_pts - 1
    for i in range(lim):
        index = []
        index.append(i)
        index.append(i+1)
        index.append(i+num_pts)
        index.append(i+num_pts+1)
        faces.append(index)
    return faces

#render bezier curves
def render_bezier_curves(bezier_curves):
    #to store info of rendered curves
    rendered_curves = {}
    #for each curve
    for curve in bezier_curves.keys():
        crv = bpy.data.curves.new('curve%d'%curve, 'CURVE')
        crv.dimensions = '3D'
        
        # make a new spline in that curve
        spline = crv.splines.new(type='NURBS')
        
        # a spline point for each point
        spline.points.add(len(bezier_curves[curve])-1) # theres already one point by default
        
        # assign the point coordinates to the spline points
        for p, new_co in zip(spline.points, bezier_curves[curve]):
            p.co = (new_co + [1.0]) # (add nurbs weight)
        
        # make a new object with the curve
        curve_name = 'Bezier Curve%d'%curve
        obj = bpy.data.objects.new(curve_name, crv)
        rendered_curves[curve] = curve_name
        bpy.context.scene.collection.objects.link(obj)        
        
    return rendered_curves

#render surface
def render_surface(verts, origin, name, num_pts):
    faces = make_faces(verts, num_pts)
    ob=createMeshFromData(name,tuple(origin),verts,faces)
    return ob

"""
Functions for generating Coons Patch
"""
#compute L value
def compute_L(pt1, pt2, param1):
    L = []
    for i in range(3):
        L.append((1-param1)*pt1[i]+param1*pt2[i])
    return L    

#compute B value
def compute_B(pt1, pt2, pt3, pt4, param1, param2):
    B = []
    for i in range(3):
        p1 = pt1[i]*(1-param1)*(1-param2)
        p2 = pt2[i]*param1*(1-param2)
        p3 = pt3[i]*(1-param1)*param2
        p4 = pt4[i]*param1*param2
        B.append(p1+p2+p3+p4)
    return B

#compute C value
def compute_C(L_c,L_d,B):
    C = []
    for i in range(3):
        C.append(L_c[i]+L_d[i]-B[i])
    return C

#compute coons patch
def compute_coons_patch_points(bezier_curves):
    #num of points in one bezier curve
    num_params = len(bezier_curves[1])
    param_lim = num_params-1
    #array to store the evaluated points
    coons_pts = []
    #list of parameters to evaluate the grid
    s_values = get_t_values(0,1,param_lim)
    t_values = get_t_values(0,1,param_lim)
    #loop and evaluate as per wiki artice. Store points in row major form
    for i in range(num_params):
        for j in range(num_params):
            L_c = compute_L(bezier_curves[1][i], bezier_curves[3][i], t_values[j])
            L_d = compute_L(bezier_curves[2][j], bezier_curves[4][j], s_values[i])
            B = compute_B(bezier_curves[1][0], bezier_curves[1][param_lim], bezier_curves[3][0], bezier_curves[3][param_lim], s_values[i], t_values[j])
            C = compute_C(L_c,L_d,B)
            coons_pts.append(C)
    #return points
    return coons_pts

#make faces for coons patch
def get_coons_patch_faces(verts,dim):
    faces = []
    for i in range(dim):
        for j in range(dim):
            index = []
            index.append(i*dim+j)
            index.append(i*dim+j+1)
            index.append((i+1)*dim+j)
            index.append((i+1)*dim+j+1)
            faces.append(index)
    #return values
    return faces

#render coons patch
def render_coons_patch(verts,faces,origin,name):
    ob=createMeshFromData(name,tuple(origin),verts,faces)
    return ob

"""
Call function as follows
1. Read control points from file
2. Determine points for bezier curves
3. Determine points for Coons patch
4. Render Coons Patch
5. Export Coons patch to external file
"""
#read and validate input points    
ctrl_pts_input_file="./bezier-control-points.txt"
ctrl_pts = read_Control_Points(ctrl_pts_input_file)
for i in range(3,5):
    ctrl_pts[i]['points'] = list(reversed(ctrl_pts[i]['points']))
#for interpolation
num_divisions = 100
#compute bezier curve points
bezier_curves = compute_bezier_curves(ctrl_pts, num_divisions)
#show bezier curves if debug enabled
debugFlag = False
if debugFlag:
    render_bezier_curves(bezier_curves)
#compute points for coons patch
coons_pts = compute_coons_patch_points(bezier_curves)
coons_faces = get_coons_patch_faces(coons_pts,num_divisions)
coons_obj = render_coons_patch(coons_pts,coons_faces,ctrl_pts[1]['points'][0],'coons-patch')
export_obj('./coons-patch.obj',coons_obj)
