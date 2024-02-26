import os
import copy
import numpy as np
from datetime import datetime
import scipy.spatial as spatial



box_size = 15                               # Radius of box in A around cluster
                                            # centre to export
cluster_file_name = "cluster_file.xyz"      # File name of the cluster
xyz_file_name = ""                          # File name of the structure
ovito_analysis = True                       # Add extra particle ID for cluster
                                            # atoms only
zero_output = True                          # Zero the output file

# Set custom bounds, or leave as None to automatically find them
bounds = np.array([102.059998, 102.059998, 102.059998])
#bounds = None

# Print XYZ file with correct atom labels
atom_type_conversion = {}
atom_type_conversion[1] = "H"
atom_type_conversion[2] = "C"





class Particle:
    
    def __init__(self, x, y, z, coordination, cluster, atom_type):
        self.position = np.array([x, y, z])           # NP[float, float, float]
        self.coordination = coordination              # Int
        self.cluster = cluster                        # Int
        self.atom_type = atom_type                    # Int


class Cluster:
    
    def __init__(self):
        self.particle_count = 0                       # Int
        self.particles = []                           # List<Particle>
        self.particle_com = np.array([0, 0, 0])       # NP[float, float, float]
        
    def AddPaticle(self, particle):
        self.particles.append(particle)
        self.particle_count += 1
        self.particle_com = self.particle_com + particle.position

    def CalculateCom(self):
        self.particle_com = self.particle_com / self.particle_count 
        


def CentrePoint(com, point, bounds, box_size):
    offset = ( (com - point) / box_size ).astype(int)
    
    offset[offset > 1] = 1
    offset[offset < -1] = -1
    
    return point + (offset * bounds)



        
        
        
# Setup       
particles = []
particles_class = []
clusters = {}
box_size_diag = ( 3 * (box_size**2) )**(1/2)
file = open(cluster_file_name, "r")
data = file.readlines()
file.close()


# Parse input file
for line in data:
    line_split = line.split(" ")
    if len(line_split) == 7:
        ID = int(line_split[0])
        atom_type = int(line_split[1])
        x = float(line_split[2])
        y = float(line_split[3])
        z = float(line_split[4])
        cluster = int(line_split[5])
        coordination = int(line_split[6])
        
        particle = Particle(x, y, z, coordination, cluster, atom_type)
        particles.append(np.array([x, y, z]))
        if (cluster not in clusters):
            clusters[cluster] = Cluster()
        clusters[cluster].AddPaticle(particle)
        particles_class.append(particle)


# Zero particles
particles = np.asarray(particles)
particles_class = np.asarray(particles_class)
mins = np.array([min(particles, key = lambda x : x[0])[0],
                 min(particles, key = lambda x : x[1])[1],
                 min(particles, key = lambda x : x[2])[2]])
for i in range(len(particles)):
    particles[i] -= mins

# Find bounds if not specified
if bounds is None:
    bounds = np.array([max(particles, key = lambda x : x[0])[0],
                       max(particles, key = lambda x : x[1])[1],
                       max(particles, key = lambda x : x[2])[2]])
    nugget = 0.00001
    bounds += np.array([nugget, nugget, nugget])


# Calculate COM
for key in range(len(clusters)):
    clusters[key + 1].CalculateCom()


# Sort clusters based on cluster size
sorted_clusters = sorted(clusters.items(), key=lambda x:x[1].particle_count, reverse=True)


# Make points tree
point_tree = spatial.cKDTree(particles, boxsize=bounds)



# Make cluster dir
save_dir = os.getcwd() + "/" + datetime.now().strftime("cluster_analysis_%Y-%m-%d_%H-%M-%S")
os.mkdir(save_dir)
os.chdir(save_dir)
cluster_rank = 1
for cluster in sorted_clusters:
    # Shift the cluster point to line up with the zeroing shift
    cluster_centre = cluster[1].particle_com - mins
    
    
    # Get all points within a sphere of point
    sphere_points = particles_class[point_tree.query_ball_point(cluster_centre, box_size_diag)]
    
    
    # Get all point in sphere within a cube or radius box_size centered at point
    box_points = []
    for sphere_point in sphere_points:
        adjusted_point = sphere_point.position
        
        # Make periodic image if atom is periodic
        if (np.linalg.norm(adjusted_point-cluster_centre) > box_size):
            adjusted_point = CentrePoint(cluster_centre, adjusted_point, bounds, box_size)
        
        # Save point if in cube within the sphere
        diff_coords = adjusted_point - cluster_centre
        if (abs(diff_coords[0]) < box_size) and (abs(diff_coords[1]) < box_size) and (abs(diff_coords[2]) < box_size):
            final_point = copy.deepcopy(sphere_point)
            final_point.position = adjusted_point
            box_points.append(final_point)
    np.asarray(box_points)
    
    
    # Save positions
    box_mins = np.array([0, 0, 0])
    if (zero_output):
        box_mins = np.array([min(box_points, key = lambda x : x.position[0]).position[0],
                             min(box_points, key = lambda x : x.position[1]).position[1],
                             min(box_points, key = lambda x : x.position[2]).position[2]])
    num_points = len(box_points)
    if ovito_analysis: num_points += len(cluster[1].particles)
    f = open("cluster_s_{}_n_{}.xyz".format(cluster_rank, cluster[0]), "w")
    f.write("{}\n{} {} {}\n".format(num_points, box_size, box_size, box_size))
    for atom in box_points:
        atom_label = atom.atom_type
        if not ovito_analysis:
            atom_label = atom_type_conversion[atom.atom_type]
        f.write("{} {} {} {}\n".format(atom_label,
                                     atom.position[0] - box_mins[0],
                                     atom.position[1] - box_mins[1],
                                     atom.position[2] - box_mins[2]))
    for atom in cluster[1].particles:
        cluster_ID = 0
        f.write("{} {} {} {}\n".format(cluster_ID,
                                     atom.position[0] - box_mins[0],
                                     atom.position[1] - box_mins[1],
                                     atom.position[2] - box_mins[2]))
    f.close()
    
    del(box_points)
    cluster_rank += 1