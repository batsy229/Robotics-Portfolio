#!/usr/bin/env python3

import rospy
import math
import cv2 as cv  # OpenCV2
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
import copy
import random
import matplotlib.cm
import scipy.io
from std_msgs.msg import Header
import struct


class Node:
    def __init__(self, x, y, idx):
        # Index of the node in the graph
        self.idx = idx

        # Position of node
        self.x = x
        self.y = y

        # Neighbouring edges
        self.neighbours = []
        self.neighbour_costs = []

        # Search parameters
        self.cost_to_node = 999999999  # A large number
        self.cost_to_node_to_goal_heuristic = 999999999  # A large number
        self.parent_node = None  # Invalid parent

    def distance_to(self, other_node):
        return math.sqrt((self.x - other_node.x) ** 2 + (self.y - other_node.y) ** 2)

    def is_connected(self, img, other_node):
        p1 = [self.x, self.y]
        p2 = [other_node.x, other_node.y]
        return not is_occluded(img, p1, p2)


class Graph:
    def __init__(self, map):

        self.map_ = map

        self.nodes_ = []

        self.grid_step_size_ = rospy.get_param("~grid_step_size")  # Grid spacing
        self.prm_num_nodes_ = rospy.get_param("~prm_num_nodes")  # Number of PRM nodes
        self.use_energy_costs_ = rospy.get_param("~use_energy_costs")  # Use the energy costs as edge costs
        self.use_prm_ = rospy.get_param("~use_prm")  # Use PRM or fixed grid

        self.groups_ = None

        # Publishers
        self.path_pub_ = rospy.Publisher('/path_planner/plan', Path, queue_size=1)
        self.path_smooth_pub_ = rospy.Publisher('/path_planner/plan_smooth', Path, queue_size=1)

        # Visualisation Marker (you can ignore this)
        self.marker_nodes_ = Marker()
        self.marker_nodes_.header.frame_id = "map"
        self.marker_nodes_.ns = "nodes"
        self.marker_nodes_.id = 0
        self.marker_nodes_.type = Marker.POINTS
        self.marker_nodes_.action = Marker.ADD
        self.marker_nodes_.pose.position.x = 0.0
        self.marker_nodes_.pose.position.y = 0.0
        self.marker_nodes_.pose.position.z = 0.0
        self.marker_nodes_.pose.orientation.x = 0.0
        self.marker_nodes_.pose.orientation.y = 0.0
        self.marker_nodes_.pose.orientation.z = 0.0
        self.marker_nodes_.pose.orientation.w = 1.0
        self.marker_nodes_.scale.x = .03
        self.marker_nodes_.scale.y = .03
        self.marker_nodes_.scale.z = .03
        self.marker_nodes_.color.a = 1.0
        self.marker_nodes_.color.r = 1.0
        self.marker_nodes_.color.g = 0.2
        self.marker_nodes_.color.b = 0.2

        self.marker_start_ = Marker()
        self.marker_start_.header.frame_id = "map"
        self.marker_start_.ns = "start"
        self.marker_start_.id = 0
        self.marker_start_.type = Marker.POINTS
        self.marker_start_.action = Marker.ADD
        self.marker_start_.pose.position.x = 0.0
        self.marker_start_.pose.position.y = 0.0
        self.marker_start_.pose.position.z = 0.0
        self.marker_start_.pose.orientation.x = 0.0
        self.marker_start_.pose.orientation.y = 0.0
        self.marker_start_.pose.orientation.z = 0.0
        self.marker_start_.pose.orientation.w = 1.0
        self.marker_start_.scale.x = .2
        self.marker_start_.scale.y = .2
        self.marker_start_.scale.z = .2
        self.marker_start_.color.a = 1.0
        self.marker_start_.color.r = 1.0
        self.marker_start_.color.g = 1.0
        self.marker_start_.color.b = 0.2

        self.marker_visited_ = Marker()
        self.marker_visited_.header.frame_id = "map"
        self.marker_visited_.ns = "visited"
        self.marker_visited_.id = 0
        self.marker_visited_.type = Marker.POINTS
        self.marker_visited_.action = Marker.ADD
        self.marker_visited_.pose.position.x = 0.0
        self.marker_visited_.pose.position.y = 0.0
        self.marker_visited_.pose.position.z = 0.0
        self.marker_visited_.pose.orientation.x = 0.0
        self.marker_visited_.pose.orientation.y = 0.0
        self.marker_visited_.pose.orientation.z = 0.0
        self.marker_visited_.pose.orientation.w = 1.0
        self.marker_visited_.scale.x = .05
        self.marker_visited_.scale.y = .05
        self.marker_visited_.scale.z = .05
        self.marker_visited_.color.a = 1.0
        self.marker_visited_.color.r = 0.2
        self.marker_visited_.color.g = 0.2
        self.marker_visited_.color.b = 1.0

        self.marker_unvisited_ = Marker()
        self.marker_unvisited_.header.frame_id = "map"
        self.marker_unvisited_.ns = "unvisited"
        self.marker_unvisited_.id = 0
        self.marker_unvisited_.type = Marker.POINTS
        self.marker_unvisited_.action = Marker.ADD
        self.marker_unvisited_.pose.position.x = 0.0
        self.marker_unvisited_.pose.position.y = 0.0
        self.marker_unvisited_.pose.position.z = 0.0
        self.marker_unvisited_.pose.orientation.x = 0.0
        self.marker_unvisited_.pose.orientation.y = 0.0
        self.marker_unvisited_.pose.orientation.z = 0.0
        self.marker_unvisited_.pose.orientation.w = 1.0
        self.marker_unvisited_.scale.x = .06
        self.marker_unvisited_.scale.y = .06
        self.marker_unvisited_.scale.z = .06
        self.marker_unvisited_.color.a = 1.0
        self.marker_unvisited_.color.r = 0.3
        self.marker_unvisited_.color.g = 1.0
        self.marker_unvisited_.color.b = 0.3

        self.marker_edges_ = Marker()
        self.marker_edges_.header.frame_id = "map"
        self.marker_edges_.ns = "edges"
        self.marker_edges_.id = 0
        self.marker_edges_.type = Marker.LINE_LIST
        self.marker_edges_.action = Marker.ADD
        self.marker_edges_.pose.position.x = 0.0
        self.marker_edges_.pose.position.y = 0.0
        self.marker_edges_.pose.position.z = 0.0
        self.marker_edges_.pose.orientation.x = 0.0
        self.marker_edges_.pose.orientation.y = 0.0
        self.marker_edges_.pose.orientation.z = 0.0
        self.marker_edges_.pose.orientation.w = 1.0
        if self.use_energy_costs_:
            # Make easier to see
            self.marker_edges_.scale.x = 0.025
            self.marker_edges_.scale.y = 0.025
            self.marker_edges_.scale.z = 0.025
        else:
            self.marker_edges_.scale.x = 0.008
            self.marker_edges_.scale.y = 0.008
            self.marker_edges_.scale.z = 0.008
        self.marker_edges_.color.a = 1.0
        self.marker_edges_.color.r = 1.0
        self.marker_edges_.color.g = 1.0
        self.marker_edges_.color.b = 0.4
        self.marker_edges_.colors = []

        self.marker_pub_ = rospy.Publisher('marker', Marker, queue_size=20)

        # Select between grid or PRM

        if self.use_prm_:
            self.create_PRM()
        else:
            self.create_grid()

        # Compute the graph connectivity
        if rospy.get_param("~show_connectivity"):
            self.find_connected_groups()

        self.visualise_graph()

    def create_grid(self):
        u = 0.1  # Rolling resistance coefficient
        m = 1025  # Mass of the rover in kg
        g = 3.71  # Gravity on Mars in m/s^2
        # Create nodes
        idx = 0
        for x in range(self.map_.min_x_, self.map_.max_x_ - 1, self.grid_step_size_):
            for y in range(self.map_.min_y_, self.map_.max_y_ - 1, self.grid_step_size_):

                if rospy.is_shutdown():
                    return

                # Check if it is occupied
                occupied = self.map_.is_occupied(x, y)

                # Create the node
                if not occupied:
                    self.nodes_.append(Node(x, y, idx))
                    idx = idx + 1

        # Create edges
        count = 0
        # distance_threshold = math.sqrt(2*(self.grid_step_size_*1.01)**2) # Chosen so that diagonals are connected, but not 2 steps away
        distance_threshold = self.grid_step_size_ * 1.01  # only 4 connected
        for node_i in self.nodes_:
            count = count + 1
            print(count, "of", len(self.nodes_))
            if rospy.is_shutdown():
                return

            for node_j in self.nodes_:

                # Don't create edges to itself
                if node_i != node_j:

                    # Check if the nodes are close to each other
                    distance = node_i.distance_to(node_j)
                    if distance < distance_threshold:

                        # Check edge is collision free
                        if node_i.is_connected(self.map_.obstacle_map_, node_j):

                            if self.use_energy_costs_:
                                # Set the edge costs as estimated energy consumption
                                ####################
                                ## YOUR CODE HERE ##
                                ## TASK 6         ##
                                ####################
                                # energy_cost = distance # Comment this out once you've done this Task
                                [x1, y1, z1] = self.map_.pixel_to_world(node_i.x, node_i.y)
                                [x2, y2, z2] = self.map_.pixel_to_world(node_j.x, node_j.y)

                                dx = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                                delta_z = z2 - z1
                                horizontal_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                                theta = math.atan2(delta_z, horizontal_distance)

                                E = abs((u * m * g * math.cos(theta) + m * g * math.sin(theta)) * dx)

                                energy_cost = E  # Set the energy cost for this edge

                            else:
                                # Define the edge cost as standard 2D Euclidean distance in pixel coordinates
                                energy_cost = distance
                            # Create the edge
                            node_i.neighbours.append(node_j)
                            node_i.neighbour_costs.append(energy_cost)

    def create_PRM(self):
        u = 0.1  # Rolling resistance coefficient
        m = 1025  # Mass of the rover in kg
        g = 3.71  # Gravity on Mars in m/s^2
        idx = 0
        num_nodes = self.prm_num_nodes_

        # Create nodes
        # hint: it will be similar to the create_grid method above

        ####################
        ## YOUR CODE HERE ##
        ## Task 7         ##
        ####################
        while idx < num_nodes:
            #x = random.randint(self.map_.min_x_, self.map_.max_x_)
            #y = random.randint(self.map_.min_y_, self.map_.max_y_)
            x = int(np.random.uniform(self.map_.min_x_, self.map_.max_x_))
            y = int(np.random.uniform(self.map_.min_y_, self.map_.max_y_))
            if rospy.is_shutdown():
                return

            occupied = self.map_.is_occupied(x, y)

            if not occupied:
                self.nodes_.append(Node(x, y, idx))
                idx += 1

        # Create edges
        count = 0
        distance_threshold = rospy.get_param("~prm_max_edge_length")
        for node_i in self.nodes_:
            count = count + 1
            print(count, "of", len(self.nodes_))
            if rospy.is_shutdown():
                return

            for node_j in self.nodes_:

                # Don't create edges to itself
                if node_i != node_j:

                    # Check if the nodes are close to each other
                    distance = node_i.distance_to(node_j)
                    if distance < distance_threshold:

                        # Check edge is collision free
                        if node_i.is_connected(self.map_.obstacle_map_, node_j):

                            if self.use_energy_costs_:

                                ############################
                                ## YOUR CODE HERE         ##
                                ## TASK 6 -- after TASK 7 ##
                                ############################
                                # energy_cost = distance # Comment this out once you've done this Task
                                # energy_cost = ??
                                [x1, y1, z1] = self.map_.pixel_to_world(node_i.x, node_i.y)
                                [x2, y2, z2] = self.map_.pixel_to_world(node_j.x, node_j.y)

                                dx = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                                delta_z = z2 - z1
                                horizontal_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                                theta = math.atan2(delta_z, horizontal_distance)

                                E = abs((u * m * g * math.cos(theta) + m * g * math.sin(theta)) * dx)

                                energy_cost = E  # Set the energy cost for this edge

                            else:

                                # Define the edge cost as standard 2D Euclidean distance in pixel coordinates
                                energy_cost = distance

                            # Create the edge
                            node_i.neighbours.append(node_j)
                            node_i.neighbour_costs.append(energy_cost)

    def get_closest_node(self, xy):
        # input: xy is a point in the form of an array, such that x=xy[0] and y=xy[1]. 
        # output: return the index of the node in self.nodes_ that has the lowest Euclidean distance to the point xy. 

        best_dist = 999999999
        best_index = None

        for i in range(len(self.nodes_)):

            ####################
            ## YOUR CODE HERE ##
            ## Task 1         ##
            ####################
            low_dist = math.sqrt((self.nodes_[i].x - xy[0]) ** 2 + (self.nodes_[i].y - xy[1]) ** 2)
            if low_dist < best_dist:
                best_dist = low_dist
                best_index = self.nodes_[i].idx

        return best_index

    def find_connected_groups(self):
        # Return a list of numbers, that has length equal to the number of nodes
        # The number in the list refers to an arbitrary "group number"
        # Two nodes should be in the same group if you can find a path from the first node to the second node

        ####################
        ## YOUR CODE HERE ##
        ## Task 8         ##
        ####################
        graph_search_obj = GraphSearch(self)

        groups = [-1] * len(self.nodes_)  # -1 indicates ungrouped nodes

        group_number = 0

        for i in range(len(self.nodes_)):
            if groups[i] == -1:
                connected_nodes = graph_search_obj.find_connected_nodes(i)

                print(f"Group {group_number + 1}: Nodes connected to node {i} -> {connected_nodes}")

                group_number += 1

                for node in connected_nodes:
                    groups[node] = group_number

        # Save the groups so they will show up in visualization
        self.groups_ = groups

    def visualise_graph(self):
        # Create and publish visualisation markers for the graph

        rospy.sleep(0.5)

        if self.groups_ == None:
            self.marker_nodes_.points = []
            for node_i in self.nodes_:
                p = self.map_.pixel_to_world(node_i.x, node_i.y)
                point = Point(p[0], p[1], p[2] + 0.1)
                self.marker_nodes_.points.append(point)
            self.marker_pub_.publish(self.marker_nodes_)
        else:
            # Plot each group with a different colour
            cmap = matplotlib.cm.get_cmap('Set1')
            colors = cmap.colors[0:-2]

            # print(self.groups_)

            self.marker_nodes_.points = []
            self.marker_nodes_.colors = []

            if rospy.get_param("~show_connectivity"):
                self.marker_nodes_.scale.x = .06
                self.marker_nodes_.scale.y = .06
                self.marker_nodes_.scale.z = .06

            for node_idx in range(len(self.nodes_)):
                # print("node_idx", node_idx)
                node_i = self.nodes_[node_idx]
                p = self.map_.pixel_to_world(node_i.x, node_i.y, )
                point = Point(p[0], p[1], p[2] + 0.1)
                self.marker_nodes_.points.append(point)

                # Set a colour
                group_id = self.groups_[node_idx]
                c = colors[group_id % len(colors)]
                col = ColorRGBA(c[0], c[1], c[2], 1.0)
                self.marker_nodes_.colors.append(col)

            self.marker_pub_.publish(self.marker_nodes_)

        rospy.sleep(0.5)

        if self.use_energy_costs_:
            # Computer upper and lower limits of energy costs
            costs_list = []
            if self.use_energy_costs_:
                for node_i in self.nodes_:
                    for node_j_index in range(len(node_i.neighbour_costs)):
                        # Use color based on the larger of i->j or j->i cost
                        cost_ij = node_i.neighbour_costs[node_j_index]
                        node_j = node_i.neighbours[node_j_index]

                        # find i in j's neighbours
                        found = False
                        for k in range(len(node_j.neighbours)):
                            if node_j.neighbours[k].idx == node_i.idx:
                                found = True
                                break

                        cost = cost_ij
                        if found:
                            cost_ji = node_j.neighbour_costs[k]
                            if cost_ji > cost_ij:
                                cost = cost_ji
                        costs_list.append(cost)
            costs_min = min(costs_list)
            costs_max = max(costs_list)

        self.marker_edges_.points = []
        self.marker_edges_.colors = []
        for node_i in self.nodes_:
            for node_j_index in range(len(node_i.neighbours)):

                node_j = node_i.neighbours[node_j_index]

                p = self.map_.pixel_to_world(node_i.x, node_i.y)
                point = Point(p[0], p[1], p[2] + 0.05)
                self.marker_edges_.points.append(point)
                p = self.map_.pixel_to_world(node_j.x, node_j.y)
                point = Point(p[0], p[1], p[2] + 0.05)
                self.marker_edges_.points.append(point)

                if self.use_energy_costs_:
                    # Use color based on the larger of i->j or j->i cost
                    cost_ij = node_i.neighbour_costs[node_j_index]
                    node_j = node_i.neighbours[node_j_index]

                    # find i in j's neighbours
                    found = False
                    for k in range(len(node_j.neighbours)):
                        if node_j.neighbours[k].idx == node_i.idx:
                            found = True
                            break

                    cost = cost_ij
                    if found:
                        cost_ji = node_j.neighbour_costs[k]
                        if cost_ji > cost_ij:
                            cost = cost_ji

                    relative_cost = ((cost - costs_min) / (costs_max - costs_min))
                    # yellow = low cost
                    # red = high cost
                    r = 1.0
                    g = (1.0 - relative_cost) ** 2.0
                    b = (1.0 - relative_cost) ** 2.0
                    # r = 1.0 * relative_cost
                    # g = ( 1.0 - relative_cost ) ** 4
                    # b = 0.0
                    # if cost < 0.5:
                    #     r = relative_cost*2
                    #     g = 1.0
                    #     b = relative_cost*2
                    # else:
                    #     r = 1.0
                    #     g = (1.0-relative_cost)*2
                    #     b = (1.0-relative_cost)*2

                    col = ColorRGBA(r, g, b, 1.0)
                    self.marker_edges_.colors.append(col)
                    self.marker_edges_.colors.append(col)

        self.marker_pub_.publish(self.marker_edges_)

        rospy.sleep(0.5)

    def visualise_search(self, visited_set, unvisited_set, start_idx, goal_idx):
        # Visualise the nodes with these node indexes
        self.marker_visited_.points = []
        for i in visited_set:
            node_i = self.nodes_[i]
            p = self.map_.pixel_to_world(node_i.x, node_i.y)
            point = Point(p[0], p[1], p[2] + 0.07)
            self.marker_visited_.points.append(point)
        self.marker_pub_.publish(self.marker_visited_)

        self.marker_unvisited_.points = []
        for i in unvisited_set:
            node_i = self.nodes_[i]
            p = self.map_.pixel_to_world(node_i.x, node_i.y)
            point = Point(p[0], p[1], p[2] + 0.07)
            self.marker_unvisited_.points.append(point)
        self.marker_pub_.publish(self.marker_unvisited_)

        self.marker_start_.points = []
        node_i = self.nodes_[start_idx]
        p = self.map_.pixel_to_world(node_i.x, node_i.y)
        point = Point(p[0], p[1], p[2] + 0.07)
        self.marker_start_.points.append(point)

        node_i = self.nodes_[goal_idx]
        p = self.map_.pixel_to_world(node_i.x, node_i.y)
        point = Point(p[0], p[1], p[2] + 0.07)
        self.marker_start_.points.append(point)
        self.marker_pub_.publish(self.marker_start_)

    def visualise_path(self, path):
        msg = Path()
        msg.header.frame_id = 'map'
        for node in path:
            p = self.map_.pixel_to_world(node.x, node.y)
            pose = PoseStamped()
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.position.z = p[2] + 0.125
            pose.pose.orientation.w = 1.0
            pose.header.frame_id = 'map'
            msg.poses.append(pose)
        self.path_pub_.publish(msg)

    def visualise_path_smooth(self, path):
        msg = Path()
        msg.header.frame_id = 'map'
        for node in path:
            p = self.map_.pixel_to_world(node.x, node.y)
            pose = PoseStamped()
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.position.z = p[2] + 0.15
            pose.pose.orientation.w = 1.0
            pose.header.frame_id = 'map'
            msg.poses.append(pose)
        self.path_smooth_pub_.publish(msg)


class Map:
    def __init__(self):

        # Extract the image from a file
        # filename = rospy.get_param('~filename')
        # self.image_ = cv.imread(filename, cv.COLOR_BGR2GRAY)

        # shape = self.image_.shape
        # self.min_x_ = 0
        # self.min_y_ = 0
        # self.max_x_ = shape[0]
        # self.max_y_ = shape[1]

        # if len(shape) == 3:
        #     self.image_ = self.image_[:,:,0]

        # Extract the map data from a file
        filename = rospy.get_param('~filename')
        mat = scipy.io.loadmat(filename)

        # mat file contains variables:
        # xs ys zs texture texture_obstacles obstacle_map
        # xs and ys are arrays containing list of x and y values of grid
        # other variables are matrices over a grid of (x,y) values
        # obstacle_map is 2D matrix of 1's (=obstacle) and 0's (=no obstacle)
        # texture and texture_obstacles have an (r,g,b) colour for each (x,y) location
        # texture is the original colour, texture_obstacles is a false colour adding obstacles in red shade

        self.x_indices_ = mat['data']['xs'][0, 0][0, :]
        self.y_indices_ = mat['data']['ys'][0, 0][0, :]
        self.terrain_height_map_ = mat['data']['zs'][0, 0]
        self.texture_ = mat['data']['texture'][0, 0]
        self.texture_obstacles_ = mat['data']['texture_obstacles'][0, 0]
        self.obstacle_map_ = mat['data']['obstacle_map'][0, 0]

        # offset the terrain map z's
        z_multiplier = 2  # to make it look more interesting
        z_offset = np.nanmin(self.terrain_height_map_)
        self.terrain_height_map_ = z_multiplier * (self.terrain_height_map_ - z_offset)

        print('obstacle_map shape', self.obstacle_map_.shape)

        self.min_x_ = 0
        self.min_y_ = 0
        self.max_x_ = len(self.x_indices_)
        self.max_y_ = len(self.y_indices_)

        self.resolution_ = rospy.get_param('~map_resolution')

        # Rviz publisher
        self.map_viz_pub_ = rospy.Publisher('terrain_viz', PointCloud2, queue_size=10)
        self.publish_rviz_map()

        # Rviz subscriber
        # self.rviz_goal_sub_ = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback, queue_size=1)
        self.rviz_goal_sub_ = rospy.Subscriber('/clicked_point', PointStamped, self.rviz_goal_callback, queue_size=1)
        self.rviz_goal_ = None

    def publish_rviz_map(self):

        points = []

        for i in range(0, self.texture_obstacles_.shape[0] - 1):
            for j in range(0, self.texture_obstacles_.shape[1] - 1):
                # extract the colour
                [r, g, b] = self.texture_obstacles_[i, j, :]  # With red colouring of obstacles
                # [r,g,b] = self.texture_[i,j,:] # Original colouring

                # extract the location
                x = self.y_indices_[i]
                y = self.x_indices_[j]
                z = self.terrain_height_map_[i, j]

                # print(x, ', ', y, ', ', z)
                # print(r, ', ', g, ', ', b)

                # append
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
                # print hex(rgb)
                pt = [x, y, z, rgb]
                points.append(pt)

        print('num points: ', len(points))

        # Assemble message
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  # PointField('rgb', 12, PointField.UINT32, 1),
                  PointField('rgba', 12, PointField.UINT32, 1),
                  ]

        header = Header()
        header.frame_id = "map"
        pc2_msg = point_cloud2.create_cloud(header, fields, points)
        pc2_msg.header.stamp = rospy.Time.now()

        # Short sleep before publishing the viz, so rviz can load
        rospy.sleep(2.)
        self.map_viz_pub_.publish(pc2_msg)
        print('published terrain cloud')

    def pixel_to_world(self, x, y):
        # return [y*self.resolution_, (self.max_x_-x)*self.resolution_]
        x = math.floor(x)
        y = math.floor(y)
        try:
            return [self.y_indices_[x], self.x_indices_[y], self.terrain_height_map_[x, y]]
        except Exception as e:
            print(e)
            print("invalid coordinates in pixel_to_world(): ", x, ", ", y)
            return [0.0, 0.0, 0.0]

    def world_to_pixel(self, x, y):
        # return [self.max_x_-(y/self.resolution_), x/self.resolution_]
        try:
            return [np.argmax(self.y_indices_ > x), np.argmax(self.x_indices_ > y)]
        except Exception as e:
            print(e)
            print("invalid coordinates in world_to_pixel(): ", x, ", ", y)
            return [0, 0]

    def rviz_goal_callback(self, msg):
        # goal = self.world_to_pixel(msg.pose.position.x, msg.pose.position.y)
        goal = self.world_to_pixel(msg.point.x, msg.point.y)
        self.rviz_goal_ = goal  # Save it into global variable
        print("New goal received from rviz!")
        print(self.rviz_goal_)

    def is_occupied(self, x, y):

        shape = self.obstacle_map_.shape

        # Out of bounds
        if x < 0 or x >= shape[0] or y < 0 or y >= shape[1]:
            return True

        if self.obstacle_map_[x, y] < 0.5:
            # Not an obstacle
            return False
        else:
            # Obstacle
            return True


def is_occluded(img, p1, p2, threshold=0.5):
    # Draws a line from p1 to p2
    # Stops at the first pixel that is a "hit", i.e. above the threshold
    # Returns the pixel coordinates for the first hit

    # Extract the vector
    x1 = float(p1[0])
    y1 = float(p1[1])
    x2 = float(p2[0])
    y2 = float(p2[1])

    step = 1.0

    dx = x2 - x1
    dy = y2 - y1
    l = math.sqrt(dx ** 2. + dy ** 2.)
    if l == 0:
        return False
    dx = dx / l
    dy = dy / l

    max_steps = int(l / step)

    for i in range(max_steps):

        # Get the next pixel
        x = int(round(x1 + dx * i))
        y = int(round(y1 + dy * i))

        # Check if it's outside the image
        if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1]:
            return False

        # Check for "hit"
        if img[x, y] >= threshold:
            return True

    # No hits found
    return False


class GraphSearch:
    def __init__(self, graph, start_xy=None, goal_xy=None):
        self.graph_ = graph
        self.path_ = []

        self.heuristic_weight_ = rospy.get_param("~heuristic_weight")

        if start_xy == None or goal_xy == None:
            # Don't do a search
            pass
        else:

            self.start_idx_ = self.graph_.get_closest_node(start_xy)
            self.goal_idx_ = self.graph_.get_closest_node(goal_xy)

            if self.start_idx_ == None or self.goal_idx_ == None:
                # Don't do a search
                pass
            else:

                if rospy.get_param("~use_naive_planner"):

                    # Do naive planner
                    self.path_ = self.naive_path_planner(self.start_idx_, self.goal_idx_)

                else:

                    # Do A* planner
                    self.search(self.start_idx_, self.goal_idx_)
                    self.path_ = self.generate_path(self.goal_idx_)

                # Visualise planned path
                self.graph_.visualise_path(self.path_)

    def naive_path_planner(self, start_idx, goal_idx):

        ####################
        ## YOUR CODE HERE ##
        ## Task 2         ##
        ####################
        path = []

        current = self.graph_.nodes_[start_idx]
        path.append(current)

        while current.idx != goal_idx:
            best_neighbour = None
            best_neighbour_distance = float('inf')
            for neighbour in current.neighbours:
                count_visits = path.count(neighbour)
                if count_visits < 3:
                    distance_to_goal = neighbour.distance_to(self.graph_.nodes_[goal_idx]) + 100 * count_visits
                    if distance_to_goal < best_neighbour_distance:
                        best_neighbour = neighbour
                        best_neighbour_distance = distance_to_goal
            if best_neighbour is None:
                # If no valid neighbours, exit
                break
            else:
                # Otherwise, move to the best neighbour
                current = best_neighbour
                path.append(current)

        return path

    def search(self, start_idx, goal_idx):

        # Set all parents and costs to zero
        for n in self.graph_.nodes_:
            n.cost_to_node = 999999999  # a large number
            n.cost_to_node_to_goal_heuristic = 999999999  # a large number
            n.parent_node = None  # invalid to begin with

        # Setup sets. These should contain indices (i.e. numbers) into the self.graph_.nodes_ array
        unvisited_set = []
        visited_set = []

        # Add start node to unvisited set
        unvisited_set.append(start_idx)
        self.graph_.nodes_[start_idx].cost_to_node = 0
        self.graph_.nodes_[start_idx].cost_to_node_to_goal_heuristic = 0

        # Loop until solution found or graph is disconnected
        while len(unvisited_set) > 0:

            # Select a node
            # hint: self.get_minimum_cost_node(unvisited_set) will help you find the node with the minimum cost

            ####################
            ## YOUR CODE HERE ##
            ## Task 3         ##
            ####################
            # Select a node with the minimum cost
            min_node_index_in_unvisited = self.get_minimum_cost_node(unvisited_set)
            node_idx = unvisited_set[min_node_index_in_unvisited]

            ####################
            ## YOUR CODE HERE ##
            ## Task 3         ##
            ####################
            # Move the node to the visited set

            visited_set.append(node_idx)
            unvisited_set.remove(node_idx)

            ####################
            ## YOUR CODE HERE ##
            ## Task 3         ##
            ####################
            # Termination criteria
            # Finish early (i.e. "return") if the goal is found

            if node_idx == goal_idx:
                rospy.loginfo("Goal found!")
                return

            # For each neighbour of the node
            for neighbour_idx in range(len(self.graph_.nodes_[node_idx].neighbours)):

                # For convenience, extract the neighbour and the edge cost from the arrays
                neighbour = self.graph_.nodes_[node_idx].neighbours[neighbour_idx]
                neighbour_edge_cost = self.graph_.nodes_[node_idx].neighbour_costs[neighbour_idx]

                # Check if neighbours is already in visited
                if neighbour.idx in visited_set:

                    # Do nothing
                    pass

                else:

                    # Compute the cost of this neighbour node
                    # hint: cost_to_node = cost-of-previous-node + cost-of-edge 
                    # hint: cost_to_node_to_goal_heuristic = cost_to_node + self.heuristic_weight_ * A*-heuristic-score
                    # hint: neighbour.distance_to() function is likely to be helpful for the heuristic-score

                    ####################
                    ## YOUR CODE HERE ##
                    ## Task 3         ##
                    ####################
                    new_cost_to_node = self.graph_.nodes_[node_idx].cost_to_node + neighbour_edge_cost
                    new_cost_to_goal_heuristic = new_cost_to_node + self.heuristic_weight_ * neighbour.distance_to(
                        self.graph_.nodes_[goal_idx])

                    # Check if neighbours is already in unvisited
                    if neighbour.idx in unvisited_set:

                        # If the cost is lower than the previous cost for this node
                        # Then update it to the new cost
                        # Also, update the parent pointer to point to the new parent 

                        ####################
                        ## YOUR CODE HERE ##
                        ## Task 3         ##
                        ####################

                        # if ??:
                        #     neighbour.parent_node = ??
                        #     neighbour.cost_to_node = ??
                        #     neighbour.cost_to_node_to_goal_heuristic = ??
                        if new_cost_to_goal_heuristic < neighbour.cost_to_node_to_goal_heuristic:
                            neighbour.parent_node = self.graph_.nodes_[node_idx]
                            neighbour.cost_to_node = new_cost_to_node
                            neighbour.cost_to_node_to_goal_heuristic = new_cost_to_goal_heuristic


                    else:

                        # Add it to the unvisited set
                        unvisited_set.append(neighbour.idx)

                        # Initialise the cost and the parent pointer
                        # hint: this will be similar to your answer above

                        ####################
                        ## YOUR CODE HERE ##
                        ## Task 3         ##
                        ####################
                        # neighbour.parent_node = ??
                        # neighbour.cost_to_node = ??
                        # neighbour.cost_to_node_to_goal_heuristic = ??
                        neighbour.parent_node = self.graph_.nodes_[node_idx]
                        neighbour.cost_to_node = new_cost_to_node
                        neighbour.cost_to_node_to_goal_heuristic = new_cost_to_goal_heuristic

            # Visualise the current search status in RVIZ
            self.visualise_search(visited_set, unvisited_set, start_idx, goal_idx)

            # Sleep for a little bit, to make the visualisation clearer
            rospy.sleep(0.01)

    def get_minimum_cost_node(self, unvisited_set):
        # Find the vertex with the minimum cost

        # There's more efficient ways of doing this...
        min_cost = 99999999
        min_idx = None
        for idx in range(len(unvisited_set)):
            cost = self.graph_.nodes_[unvisited_set[idx]].cost_to_node_to_goal_heuristic
            if cost < min_cost:
                min_cost = cost
                min_idx = idx
        return min_idx

    def generate_path(self, goal_idx):
        # Generate the path by following the parents from the goal back to the start

        path = []

        current = self.graph_.nodes_[goal_idx]
        path.append(current)

        ####################
        ## YOUR CODE HERE ##
        ## Task 4         ##
        ####################
        while current.parent_node is not None:
            current = current.parent_node
            path.append(current)
        path.reverse()

        return path

    def find_connected_nodes(self, start_idx):
        # Return a list of all nodes that are reachable from start_idx node
        # Hint 1 : this should be very similar to search(self, start_idx, goal_idx)
        # Except there's no goal_idx, and it returns a list of all node indices that have a valid path from start_idx
        # Hint 2: Can we use A* heuristic if there's no goal?

        #pass  # added here to avoid whitespace issue

        ####################
        ## YOUR CODE HERE ##
        ## Task 8         ##
        ####################
        for n in self.graph_.nodes_:
            n.cost_to_node = float('inf')  # A large number to represent "unvisited"
            n.parent_node = None  # Reset parent to None

            # Setup unvisited and visited sets
        unvisited_set = []
        visited_set = []

        # Add the start node to the unvisited set
        unvisited_set.append(start_idx)
        self.graph_.nodes_[start_idx].cost_to_node = 0  # Cost to reach start node is 0

        # Loop until there are no more nodes in the unvisited set
        while len(unvisited_set) > 0:

            # Select the node with the minimum cost (Dijkstra's logic for selecting the closest node)
            min_node_index_in_unvisited = self.get_minimum_cost_node(unvisited_set)
            if min_node_index_in_unvisited is None:  # If no valid node found, exit loop
                break
            node_idx = unvisited_set[min_node_index_in_unvisited]

            # Move the selected node to the visited set
            visited_set.append(node_idx)
            unvisited_set.remove(node_idx)

            # Explore neighbors of the current node
            for neighbour_idx in range(len(self.graph_.nodes_[node_idx].neighbours)):
                neighbour = self.graph_.nodes_[node_idx].neighbours[neighbour_idx]
                neighbour_edge_cost = self.graph_.nodes_[node_idx].neighbour_costs[neighbour_idx]

                # If the neighbor is already visited, skip it
                if neighbour.idx in visited_set:
                    continue

                # Calculate the new cost to reach the neighbor
                new_cost_to_node = self.graph_.nodes_[node_idx].cost_to_node + neighbour_edge_cost

                # If the neighbor is already in unvisited, check if the new cost is lower
                if neighbour.idx in unvisited_set:
                    if new_cost_to_node < neighbour.cost_to_node:
                        # Update the cost and parent node
                        neighbour.cost_to_node = new_cost_to_node
                        neighbour.parent_node = self.graph_.nodes_[node_idx]
                else:
                    # If neighbor is not in the unvisited set, add it and update its cost and parent
                    unvisited_set.append(neighbour.idx)
                    neighbour.cost_to_node = new_cost_to_node
                    neighbour.parent_node = self.graph_.nodes_[node_idx]

        # Return all visited nodes as a list (i.e., all nodes connected to start_idx)
        return visited_set


    def visualise_search(self, visited_set, unvisited_set, start_idx, goal_idx):
        self.graph_.visualise_search(visited_set, unvisited_set, start_idx, goal_idx)


class PathSmoother():
    def __init__(self, graph, path):
        self.graph_ = graph
        self.path_ = self.smooth_path(path)
        self.graph_.visualise_path_smooth(self.path_)

    def smooth_path(self, path_nodes):

        # Convert into into a geometry_msgs.Point[]
        path = []

        for node in path_nodes:
            p = Point()
            p.x = node.x
            p.y = node.y
            path.append(p)

        # Initialise the smooth path
        path_smooth = copy.deepcopy(path)

        alpha = rospy.get_param("~alpha")
        beta = rospy.get_param("~beta")

        # Loop until the smoothing converges
        # In each iteration, update every waypoint except the first and last waypoint

        ####################
        ## YOUR CODE HERE ##
        ## Task 5         ##
        ####################
        tolerance = 0.001
        difference = tolerance
        while difference >= tolerance:
            difference = 0.0
            path_smooth_new = copy.deepcopy(path_smooth)
            for i in range(1, len(path_smooth) - 1):
                # Update the smooth path using the formula
                path_smooth_new[i].x = path_smooth[i].x - ((alpha + 2 * beta) * path_smooth[i].x) + \
                                       (alpha * path[i].x) + (beta * path_smooth[i - 1].x) + (
                                                   beta * path_smooth[i + 1].x)
                path_smooth_new[i].y = path_smooth[i].y - ((alpha + 2 * beta) * path_smooth[i].y) + \
                                       (alpha * path[i].y) + (beta * path_smooth[i - 1].y) + (
                                                   beta * path_smooth[i + 1].y)
                if is_occluded(self.graph_.map_.obstacle_map_, [path_smooth_new[i - 1].x, path_smooth_new[i - 1].y],
                               [path_smooth_new[i].x, path_smooth_new[i].y]) \
                        or is_occluded(self.graph_.map_.obstacle_map_, [path_smooth_new[i].x, path_smooth_new[i].y],
                                       [path_smooth_new[i + 1].x, path_smooth_new[i + 1].y]):
                    path_smooth_new[i] = copy.deepcopy(path_smooth[i])
            for i in range(len(path_smooth)):
                difference += (path_smooth_new[i].x - path_smooth[i].x) ** 2 + (
                            path_smooth_new[i].y - path_smooth[i].y) ** 2
            path_smooth = copy.deepcopy(path_smooth_new)

        return path_smooth


if __name__ == '__main__':
    # Create the ROS node
    rospy.init_node('path_planner')

    # Create a map from image
    map = Map()

    # Sleep added to give rviz a chance to wake up
    rospy.sleep(1.0)

    # Create a graph from the map
    graph = Graph(map)

    if not rospy.get_param("~show_connectivity"):

        startx = rospy.get_param("~startx")
        starty = rospy.get_param("~starty")
        goalx = rospy.get_param("~goalx")
        goaly = rospy.get_param("~goaly")

        # Do the graph search
        graph_search = GraphSearch(graph, [startx, starty], [goalx, goaly])

        # Smooth the path
        PathSmoother(graph, graph_search.path_)

        print("Plan finished! Click a new goal in rviz 2D Nav Goal.")

        # Re-plan indefinitely when rviz goals received
        while not rospy.is_shutdown():

            if map.rviz_goal_ is None:
                # Do nothing, waiting for goal
                rospy.sleep(0.01)
            else:

                # Extract the goal
                startx = goalx
                starty = goaly
                goalx, goaly = map.rviz_goal_
                map.rviz_goal_ = None  # Clear it so a new goal can be set

                # Do the graph search
                graph_search = GraphSearch(graph, [startx, starty], [goalx, goaly])

                # Smooth the path
                PathSmoother(graph, graph_search.path_)

                print("Plan finished! Click a new goal in rviz 2D Nav Goal.")

    # Loop forever while processing callbacks
    rospy.spin()
