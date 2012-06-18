#! /usr/bin/env python

import math, random, re, os, sys
import pygame

epsilon = sys.float_info.epsilon

class Point2D():
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    
    def __lt__(self, other):
        if self.x < other.x:
            return True
        if self.y < other.y:
            return True
        return False
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __str__(self):
        return "Point(%.2f, %.2f)" % (self.x, self.y)
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def distance_to_point(self, other):
        return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2)


class Vector2D():
    def __init__(self, A, B):
        # a vector can be created from two points or by directly setting values
        if isinstance(A, Point2D) and isinstance(B, Point2D):
            self.A = A
            self.B = B
            self.x = B.x-A.x
            self.y = B.y-A.y
        else:
            self.A = None
            self.B = None
            self.x = A
            self.y = B
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __str__(self):
        return "Vector(%.2f, %.2f)" % (self.x, self.y)
    
    def dot(self, other):
        return self.x*other.x+self.y*other.y
    
    def get_orthogonal(self):
        # returned vector will always point upwards
        i = -1 if self.x < 0 else 1
        return Vector2D(-i*self.y, i*self.x)


class Line2D():
    def __init__(self, A, AB):
        if isinstance(A, Point2D) and isinstance(AB, Vector2D):
            self.anchor = A
            self.vector = AB
    
    def __call__(self, k):
        return Point2D(self.anchor.x+k*self.vector.x, self.anchor.y+k*self.vector.y)
    
    def __str__(self):
        return "Line(anchor=%s, vector=%s)" % (self.anchor, self.vector)


class Circle2D():
    def __init__(self, C, r):
        if isinstance(C, Point2D):
            self.centre = C
            self.radius = float(r)
    
    def __str__(self):
        return "Circle(centre=%s, radius=%.2f)" % self.centre
    
    def contains(self, point):
        return ((point.x-self.centre.x)**2 + (point.y-self.centre.y)**2) < (self.radius**2 + epsilon)


def process_cli(args):
    # default values
    init_centre_strategy = 0
    use_random_data = True
    num_pnt, num_cnt = 337, 17
    sc_x, sc_y, point_size = 10, 10, 2
    win_w, win_h = 800, 600
    
    # command line switches
    flags = {
        "-?":{"long":"--help",    "arg":"",    "desc":"Show this help message and exit."},
        "-k":{"long":"--keys",    "arg":"",    "desc":"Show in-simulator keybindings and exit."},
        "-d":{"long":"--data",    "arg":"P",   "desc":"Use data found at P."},
        "-w":{"long":"--width",   "arg":"N",   "desc":"Set window width to N (default: %d)" % win_w},
        "-h":{"long":"--height",  "arg":"N",   "desc":"Set window height to N (default: %d)" % win_h},
        "-s":{"long":"--scaling", "arg":"N",   "desc":"Set zoom factor to N (default: %d)" % sc_x},
        "-p":{"long":"--points",  "arg":"N",   "desc":"Generate N random data points (default: %d)" % num_pnt},
        "-c":{"long":"--centres", "arg":"N",   "desc":"Generate N random centres (default: %d)" % num_cnt},
        "-r":{"long":"--radius",  "arg":"N",   "desc":"Set point radius to N (default: %d)" % point_size},
        "-i":{"long":"--init",    "arg":"N",   "desc":"Centre-choice: random 0, extrema 1, low 2, high 3"}}
    flags_short = ["-?", "-k", "-d", "-w", "-h", "-s", "-p", "-c", "-r", "-i"]
    flag_str = ["[%s%s]" % (f, "" if flags[f]["arg"] == "" else " " + flags[f]["arg"]) for f in flags_short]
    
    # process command line arguments
    if len(args) > 1:
        i = 1
        while i < len(args):
            p = args[i]
            if p == "--help" or p == "-?":
                print "Usage:\t%s %s" % (args[0], " ".join(flag_str))
                print "\nVisalises the k-means clustering algorithm.\n"
                print "Optional arguments:"
                for f in flags_short:
                    arg = "" if flags[f]["arg"] == "" else " " + flags[f]["arg"]
                    str = "%s%s, %s%s" % (f, arg, flags[f]["long"], arg)
                    print str + " "*(20-len(str)) + flags[f]["desc"]
                sys.exit(0)
            elif p == "--keys" or p == "-k":
                print "Key bindings for k-means clustering algorithm visualiser.\n"
                for key in key_bindings:
                    print "\t key %s -> %s" % (chr(key_bindings[key]), key)
                sys.exit(0)
            elif p in flags_short and i == len(args)-1:
                sys.exit("\"%s\" flag should be followed B.y a%s." % (p, "n integer" if flags[p]["arg"] == "N" else " string"))
            elif p == "--data" or p == "-d":
                path = args[i+1]
                try:
                    # read data from file and filter out any non digit values
                    file = re.sub("[^0-9]", " ", open(path).read())
                except IOError:
                    sys.exit("Couldn't find %s" % path)
                # convert data to Points2D
                data = [int(x) for x in re.split("[ ]+", file) if x != "" ]
                data = [Point2D(x, y) for (x, y) in zip(data[::2], data[1::2])]
                use_random_data = False
                num_pnt = len(data)
            elif p in flags_short and not args[i+1].isdigit():
                sys.exit("\"%s\" flag should be followed B.y an integer." % p)
            elif p == "--width" or p == "-w":
                win_w = int(args[i+1])
            elif p == "--height" or p == "-h":
                win_h = int(args[i+1])
            elif p == "--scaling" or p == "-s":
                sc_x = sc_y = int(args[i+1])
            elif p == "--points" or p == "-p":
                num_pnt = int(args[i+1])
            elif p == "--centres" or p == "-c":
                num_cnt = int(args[i+1])
            elif p == "--radius" or p == "-r":
                point_size = int(args[i+1])
            elif p == "--init" or p == "-i":
                if args[i+1] in "0123":
                    init_centre_strategy = int(args[i+1])
                else:
                    sys.exit("%s flag takes arguments 0/1/2/3 only (provided: %s)." % (p, args[i+1]))
            else:
                sys.exit("Unknown flag \"%s\". Try \"%s --help\" for documentation." % (p, args[0]))
            i += 2
    
    if use_random_data:
        # generate random data points
        data = [Point2D(x, y) for x in xrange(win_w/sc_x+1) for y in xrange(win_h/sc_y+1)]
        if num_cnt > len(data) or num_pnt > len(data):
            cause = "points" if num_pnt > len(data) else "centres"
            sys.exit("Too many %s for scaling factor x=%d y=%d (mA.x=%d)." % (cause, sc_x, sc_y, len(data)))
        data = random.sample(data, num_pnt)
    else:
        # adjust scaling-factor to fit data into window
        xs, ys = sorted([point.x for point in data]), sorted([point.y for point in data])
        dx, dy = -xs[0] if xs[0] < 0 else 0, -ys[0] if ys[0] < 0 else 0
        data = [Point2D(point.x+dx, point.y+dy) for point in data]
        sc_x = (win_w*0.975)/max(xs[-1]+dx, 1)
        sc_y = (win_h*0.975)/max(ys[-1]+dy, 1)
        point_size = min(sc_x, sc_y)/10
    
    if init_centre_strategy == 0:
        # use random points as initial cluster-centres
        centres = random.sample(data, num_cnt)
    elif init_centre_strategy == 1:
        # use points with most extreme values as initial cluster-centres
        sorted_data = sorted(data)
        centres = sorted_data[:num_cnt/2] + sorted_data[-num_cnt/2:]
    elif init_centre_strategy == 2 or init_centre_strategy == 3:
        # use points with highest deviation from others as initial cluster-centres
        xs = [point.x for point in data]
        ys = [point.y for point in data]
        m_x = float(sum(xs))/len(xs)
        m_y = float(sum(ys))/len(ys)
        s_x = math.sqrt(float(sum([(x-m_x)**2 for x in xs]))/(len(xs)-1))
        s_y = math.sqrt(float(sum([(x-m_y)**2 for x in ys]))/(len(ys)-1))
        st_xs = sorted([((xs[i]-m_x)/s_x, i) for i in range(len(xs))])
        st_ys = sorted([((ys[i]-m_y)/s_y, i) for i in range(len(ys))])
        # use points with largest deviation in low direction
        if init_centre_strategy == 2:
            xs = [data[i] for (deviation, i) in st_xs][:int(math.ceil(num_cnt/2.0))]
            ys = [data[i] for (deviation, i) in st_ys if data[i] not in xs][:int(math.floor(num_cnt/2.0))]
        # use points with largest deviation in high direction
        elif init_centre_strategy == 3:
            xs = [data[i] for (deviation, i) in st_xs][-int(math.ceil(num_cnt/2.0)):]
            ys = [data[i] for (deviation, i) in st_ys if data[i] not in xs][-int(math.floor(num_cnt/2.0)):]
        centres = xs + ys
    
    return data, centres, win_w, win_h, (sc_x, sc_y), point_size

def draw_line_scaled(surface, col, origin, (sc_x, sc_y), start, end, width=1):
    scaled_start = (int(origin.x+sc_x*start.x), int(origin.y-sc_y*start.y))
    scaled_end = (int(origin.x+sc_x*end.x), int(origin.y-sc_y*end.y))
    pygame.draw.line(surface, col, scaled_start, scaled_end, int(width))

def draw_circle_scaled(surface, col, origin, (sc_x, sc_y), centre, radius, width=0):
    scaled_centre = (int(origin.x+centre.x*sc_x), int(origin.y-centre.y*sc_y))
    pygame.draw.circle(surface, col, scaled_centre, int(radius), int(width))

def cluster_lines(centres, clusters):
    return [(centres[i], point) for i in range(len(clusters)) for point in clusters[i]]

def compute_mean_point(points):
    xs, ys = [point.x for point in points], [point.y for point in points]
    return Point2D(float(sum(xs))/len(points), float(sum(ys))/len(points))

def update_clusters(centres, data):
    clusters = [[] for centre in centres]
    dist_points_to_centres = [[point.distance_to_point(cntr) for cntr in centres] for point in data]
    for i in range(len(dist_points_to_centres)):
        min_value, min_value_index = dist_points_to_centres[i][0], 0
        for j in range(1, len(dist_points_to_centres[i])):
            if dist_points_to_centres[i][j] < min_value:
                min_value, min_value_index = dist_points_to_centres[i][j], j
        clusters[min_value_index].append(data[i])
    return clusters

def iterate_k_means(centres, data, clusters):
    clusters = update_clusters(centres, data)
    centres = map(compute_mean_point, clusters)
    return (centres, clusters)

def circumcircle(A, B, C):
    # precondition: points not collinear
    D = 2*(A.x*(B.y-C.y)+B.x*(C.y-A.y)+C.x*(A.y-B.y))
    Ux = ((A.x**2+A.y**2)*(B.y-C.y)+(B.x**2+B.y**2)*(C.y-A.y)+(C.x**2+C.y**2)*(A.y-B.y))/D
    Uy = ((A.x**2+A.y**2)*(C.x-B.x)+(B.x**2+B.y**2)*(A.x-C.x)+(C.x**2+C.y**2)*(B.x-A.x))/D
    centre = Point2D(Ux, Uy)
    radius = centre.distance_to_point(A)
    return Circle2D(centre, radius)

def collinear(A, B, C):
    return -epsilon <= ((B.x-A.x)*(C.y-A.y)-(C.x-A.x)*(B.y-A.y)) <= epsilon

def midpoint(A, B):
    return Point2D((A.x+B.x)/2.0, (A.y+B.y)/2.0)

def bisector(A, B):
    mid = midpoint(A, B)
    if -epsilon <= (B.x-A.x) <= epsilon:
        return lambda k: Point2D(k, mid.y)
    if -epsilon <= (B.y-A.y) <= epsilon:
        return lambda k: Point2D(mid.x, k)
    m = -(B.x-A.x)/(B.y-A.y)
    b = mid.y - m*mid.x
    return lambda k: Point2D(k, m*k+b)

def voronify(delaunay_nodes, delaunay_faces):
    # precondition: access to delaunay-triangulation of polygon
    if len(delaunay_nodes) <= 1:
        return (delaunay_nodes, [])
    if len(delaunay_nodes) == 2:
        fst, snd = delaunay_nodes[0], delaunay_nodes[1]
        perp = bisector(fst, snd)
        return ([midpoint(fst, snd)], [(perp(1000), perp(-1000))])
    
    nodes = []
    edges = []
    graph = {}
    spanning = {}
    
    # build graph
    for face in delaunay_faces:
        (A, B, C) = face
        circumcentre = circumcircle(A, B, C).centre
        nodes.append(circumcentre)
        graph[face] = {
            "circumcentre":circumcentre,
            "edges":{(A, B):[face], (A, C):[face], (B, C):[face]},
            "neighbours":[] }
        for (M, N, O) in delaunay_faces:
            # if any two out of three vertices of the triangles match then the triangles are neighbours
            if A==M and B==N and C==O: continue
            elif (A==M and (B==N or B==O)) or (A==N and (B==M or B==O)) or (A==O and (B==M or B==N)):
                graph[(A, B, C)]["neighbours"].append((M, N, O))
                graph[(A, B, C)]["edges"][(A, B)].append((M, N, O))
            elif (B==M and (A==N or A==O)) or (B==N and (A==M or A==O)) or (B==O and (A==M or A==N)):
                graph[(A, B, C)]["neighbours"].append((M, N, O))
                graph[(A, B, C)]["edges"][(A, B)].append((M, N, O))
            elif (A==M and (C==N or C==O)) or (A==N and (C==M or C==O)) or (A==O and (C==M or C==N)):
                graph[(A, B, C)]["neighbours"].append((M, N, O))
                graph[(A, B, C)]["edges"][(A, C)].append((M, N, O))
            elif (C==M and (A==N or A==O)) or (C==N and (A==M or A==O)) or (C==O and (A==M or A==N)):
                graph[(A, B, C)]["neighbours"].append((M, N, O))
                graph[(A, B, C)]["edges"][(A, C)].append((M, N, O))
            elif (B==M and (C==N or C==O)) or (B==N and (C==M or C==O)) or (B==O and (C==M or C==N)):
                graph[(A, B, C)]["neighbours"].append((M, N, O))
                graph[(A, B, C)]["edges"][(B, C)].append((M, N, O))
            elif (C==M and (B==N or B==O)) or (C==N and (B==M or B==O)) or (C==O and (B==M or B==N)):
                graph[(A, B, C)]["neighbours"].append((M, N, O))
                graph[(A, B, C)]["edges"][(B, C)].append((M, N, O))
        if len(graph[face]["neighbours"]) <= 2:
            # face has two or fewer neighbours - can only happen if it is unbounded on some side(s)
            spanning[face] = {"edges":[], "circumcentre":circumcentre}
            for edge in graph[face]["edges"]:
                if len(graph[face]["edges"][edge]) == 1:
                    # add any edges with only one bounded neighbouring face to spanning polygon
                    spanning[face]["edges"].append(edge)
    
    for face in graph:
        neighbours = graph[face]["neighbours"]
        circumcentre = graph[face]["circumcentre"]
        
        # connect circumcentres of adjacent faces
        for neighbour in neighbours:
            edges.append((graph[neighbour]["circumcentre"], circumcentre))
        
        # if face is unbounded on some side: also need to draw rays pointing outwards
        if face in spanning:
            for edge in spanning[face]["edges"]:
                A, B = edge
                mid = midpoint(A, B)
                AB = Vector2D(A, B)
                perp = Line2D(circumcentre, AB.get_orthogonal())
                
                # two cases: if circumcentre is inside polygon, draw ray pointing towards edge
                # otherwise: draw ray pointing away from edge
                if in_polygon(circumcentre, delaunay_faces):
                    if area_of_triangle(perp(1), A, B) < area_of_triangle(perp(-1), A, B):
                        edges.append((circumcentre, perp(1000)))
                    else:
                        edges.append((circumcentre, perp(-1000)))
                else:
                    if area_of_triangle(perp(1), A, B) > area_of_triangle(perp(-1), A, B):
                        edges.append((circumcentre, perp(1000)))
                    else:
                        edges.append((circumcentre, perp(-1000)))
    return (nodes, edges)

def area_of_triangle(A, B, C):
    return abs((A.x-C.x)*(B.y-A.y)-(A.x-B.x)*(C.y-A.y))/2.0

def in_polygon(point, triangles):
    # precondition: polygon triangulated
    for triangle in triangles:
        if in_triangle(point, triangle):
            return True
    return False

def in_triangle(P, (A, B, C)):
    # breaks for some triangles no matter which approach is used
    return in_triangle1(P, (A, B, C))

def in_triangle1(P, (A, B, C)):
    # http://www.blackpawn.com/texts/pointinpoly/default.html
    v0 = Vector2D(A, C)
    v1 = Vector2D(A, B)
    v2 = Vector2D(A, P)
    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)
    inv_denom = 1.0 / (dot00*dot11-dot01*dot01)
    u = (dot11*dot02-dot01*dot12)*inv_denom
    v = (dot00*dot12-dot01*dot02)*inv_denom
    return u >= -epsilon and v >= -epsilon and (u+v) <= (1+epsilon)

def in_triangle2(P, (A, B, C)):
    b1 = sign(P, A, B) <= epsilon
    b2 = sign(P, B, C) <= epsilon
    b3 = sign(P, C, A) <= epsilon
    return (b1 == b2) and (b2 == b3)

def in_triangle3(P, (A, B, C)):
    # http://www.visibone.com/inpoly/
    npoints = 3
    xt, yt = P.x, P.y
    poly = [[A.x, A.y], [B.x, B.y], [C.x, C.y]]
    inside = False
    xold = poly[npoints-1][0]
    yold = poly[npoints-1][1]
    for i in range(npoints):
        xnew = poly[i][0]
        ynew = poly[i][1]
        if xnew > xold:
            x1 = xold
            x2 = xnew
            y1 = yold
            y2 = ynew
        else:
            x1 = xnew
            x2 = xold
            y1 = ynew
            y2 = yold
        if (((xnew < xt) == (xt <= xold)) and ((yt-y1)*(x2-x1) < (y2-y1)*(xt-x1))):
            inside = not inside
        xold = xnew
        yold = ynew
    return inside

def sign(A, B, C):
    return (A.x-C.x)*(B.y-C.y)-(B.x-C.x)*(A.y-C.y)

def any_in_circumcircle(points, (A, B, C)):
    for i in range(len(points)):
        P = points[i]
        if (P == A or P == B or P == C):
            continue
        if collinear(A, B, C) or circumcircle(A, B, C).contains(P):
            return True
    return False

def delaunay_triangulation(V):
    if len(V) == 1:
        return (V, [], [])
    if len(V) == 2:
        return (V, [(V[0], V[1])], [])
    
    # brute force algorithm = O(n^4)
    nodes, edges, faces = [], [], []
    for i in range(len(V)-2):
        for j in range(i+1, len(V)-1):
            for k in range(j+1, len(V)):
                if not any_in_circumcircle(V, (V[i], V[j], V[k])):
                    nodes.extend([V[i], V[j], V[k]])
                    edges.extend([(V[i], V[j]), (V[i], V[k]), (V[j], V[k])])
                    faces.append((V[i], V[j], V[k]))
    return (nodes, edges, faces)

if __name__ == "__main__":
    # constants values
    back_col = [ 37,  37,  37]
    data_col = [ 34, 139,  34]
    cent_col = [208,  32, 144]
    grid_col = [100, 100, 100]
    line_col = [205, 200, 177]
    tria_col = [100, 149, 237]
    voro_col = [255, 140,   0]
    key_bindings = {    "save to file":pygame.K_s,
                        "next iteration":pygame.K_n,
                        "toggle grid":pygame.K_g,
                        "toggle cluster lines":pygame.K_c,
                        "toggle triangulation":pygame.K_t,
                        "toggle voronoi":pygame.K_v,
                        "exit":pygame.K_ESCAPE}
    
    # set up data
    data, centres, win_w, win_h, scale, point_size = process_cli(sys.argv)
    clusters = update_clusters(centres, data)
    origin = Point2D(0, win_h)
    delaunay_nodes, delaunay_edges, delaunay_faces = delaunay_triangulation(centres)
    voronoi_nodes, voronoi_edges = voronify(delaunay_nodes, delaunay_faces)
    membership_lines = cluster_lines(centres, clusters)
    grid_ver = [(Point2D(i, 0), Point2D(i, int(origin.y))) for i in range(int(origin.x), win_w+1)]
    grid_hor = [(Point2D(int(origin.x), i), Point2D(win_w, i)) for i in range(int(origin.y), 0, -1)]
    grid = grid_hor + grid_ver
    num_pnt = len(data)
    num_cnt = len(centres)
    
    # set up pygame
    os.environ["SDL_VIDEO_CENTERED"] = "1"    
    pygame.init()
    screen, clock = pygame.display.set_mode((win_w, win_h)), pygame.time.Clock()
    pygame.display.set_caption("k-means visualiser with %d points and %d centres" % (num_pnt, num_cnt))
    
    # display loop
    show_membership_lines = True
    show_grid, show_triangulation, show_voronoi = False, False, False
    converged, iters = False, 0
    redraw = True
    done = False
    while not done:
        if redraw:
            redraw = False
            screen.fill(back_col)
            
            # draw grid
            if show_grid:
                for (p1, p2) in grid:
                    draw_line_scaled(screen, grid_col, origin, scale, p1, p2)
            
            # draw cluster-membership lines
            if show_membership_lines:
                for (p1, p2) in membership_lines:
                    draw_line_scaled(screen, line_col, origin, scale, p1, p2)
            
            # draw triangulation
            if show_triangulation:
                for (A, B) in delaunay_edges:
                    draw_line_scaled(screen, tria_col, origin, scale, A, B)
            
            # draw voronoi decision boundaries
            if show_voronoi:
                for (A, B) in voronoi_edges:
                    draw_line_scaled(screen, voro_col, origin, scale, A, B)
                for P in voronoi_nodes:
                    draw_circle_scaled(screen, voro_col, origin, scale, P, point_size*3, point_size)
            
            # draw points
            for point in data:
                draw_circle_scaled(screen, data_col, origin, scale, point, point_size)
            
            # draw centres
            for centre in centres:
                draw_circle_scaled(screen, cent_col, origin, scale, centre, point_size*5, point_size*2)
        
        # process user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == key_bindings["save to file"]:
                    pygame.image.save(screen, "k_means_points=%d_centres=%d_runs=%d.png" % (num_pnt, num_cnt, iters))
                elif event.key == key_bindings["next iteration"] and not converged:
                    old_centres = centres
                    (centres, clusters) = iterate_k_means(centres, data, clusters)
                    membership_lines = cluster_lines(centres, clusters)
                    delaunay_nodes, delaunay_edges, delaunay_faces = delaunay_triangulation(centres)
                    voronoi_nodes, voronoi_edges = voronify(delaunay_nodes, delaunay_faces)
                    iters = iters+1
                    pygame.display.set_caption("k-means: %d points, %d centres, %d runs" % (num_pnt, num_cnt, iters))
                    if old_centres == centres and not converged:
                        converged = True
                        pygame.display.set_caption("%s - Converged!" % (pygame.display.get_caption()[0]))
                    redraw = True
                elif event.key == key_bindings["toggle grid"]:
                    show_grid = not show_grid
                    redraw = True
                elif event.key == key_bindings["toggle cluster lines"]:
                    show_membership_lines = not show_membership_lines
                    redraw = True
                elif event.key == key_bindings["toggle triangulation"]:
                    show_triangulation = not show_triangulation
                    redraw = True
                elif event.key == key_bindings["toggle voronoi"]:
                    show_voronoi = not show_voronoi
                    redraw = True
                elif event.key == key_bindings["exit"]:
                    done = True
        
        clock.tick(20)
        pygame.display.flip()
    pygame.quit()
