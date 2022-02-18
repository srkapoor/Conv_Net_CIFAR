import numpy as np
from canny import *

"""
   INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)

   Implement an interest point operator of your choice.

   Your operator could be:

   (A) The Harris corner detector (Szeliski 4.1.1)

               OR

   (B) The Difference-of-Gaussians (DoG) operator defined in:
       Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
       https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

               OR

   (C) Any of the alternative interest point operators appearing in
       publications referenced in Szeliski or in lecture

              OR

   (D) A custom operator of your own design

   You implementation should return locations of the interest points in the
   form of (x,y) pixel coordinates, as well as a real-valued score for each
   interest point.  Greater scores indicate a stronger detector response.

   In addition, be sure to apply some form of spatial non-maximum suppression
   prior to returning interest points.

   Whichever of these options you choose, there is flexibility in the exact
   implementation, notably in regard to:

   (1) Scale

       At what scale (e.g. over what size of local patch) do you operate?

       You may optionally vary this according to an input scale argument.

       We will test your implementation at the default scale = 1.0, so you
       should make a reasonable choice for how to translate scale value 1.0
       into a size measured in pixels.

   (2) Nonmaximum suppression

       What strategy do you use for nonmaximum suppression?

       A simple (and sufficient) choice is to apply nonmaximum suppression
       over a local region.  In this case, over how large of a local region do
       you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.

   If you detect more interest points than the requested maximum (given by
   the max_points argument), return only the max_points highest scoring ones.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy array
      max_points  - maximum number of interest points to return
      scale       - (optional, for your use only) scale factor at which to
                    detect interest points

   Returns:
      xs          - numpy array of shape (N,) containing x-coordinates of the
                    N detected interest points (N <= max_points)
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing a real-valued
                    measurement of the relative strength of each interest point
                    (e.g. corner detector criterion OR DoG operator magnitude)
"""
def find_interest_points(image, max_points = 200, scale = 1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   I_x, I_y = sobel_gradients(image)
   #convolving with gaussian to correspond to outer products of sobel
   I_xx = conv_2d_gaussian(I_x**2, scale)
   I_yy = conv_2d_gaussian(I_y**2, scale)
   I_xy = conv_2d_gaussian(I_x * I_y, scale)
   #finding the harris response by calculating determinant and trace
   detA = I_xx * I_yy - I_xy**2
   traceA = I_xx + I_yy
   harris_response = detA - 0.06*traceA**2
   #finding the gradient direction
   theta = np.arctan2(I_y, I_x)
   #conducting nonmax suppression to find local maximums of harris response
   nonmax = nonmax_suppress(harris_response, theta)
   indices = []
   for i in range(nonmax.shape[0]):
       for j in range(nonmax.shape[1]):
           #keeping track of x,y coordiantes and their score in a list of tuples
           indices.append((i, j, nonmax[i,j]))
   #sorting the tuple
   indices.sort(key = lambda x: x[2])
   #finding only the required number of max points
   interest = indices[-max_points:]
   #outputting the elements in the tuple inidividually
   xs = np.asarray([inttup[0] for inttup in interest])
   ys = np.asarray([inttup[1] for inttup in interest])
   scores = np.asarray([inttup[2] for inttup in interest])
   return xs, ys, scores

"""
   FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)

   Implement a SIFT-like feature descriptor by binning orientation energy
   in spatial cells surrounding an interest point.

   Unlike SIFT, you do not need to build-in rotation or scale invariance.

   A reasonable default design is to consider a 3 x 3 spatial grid consisting
   of cell of a set width (see below) surrounding an interest point, marked
   by () in the diagram below.  Using 8 orientation bins, spaced evenly in
   [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions.

             ____ ____ ____
            |    |    |    |
            |    |    |    |
            |____|____|____|
            |    |    |    |
            |    | () |    |
            |____|____|____|
            |    |    |    |
            |    |    |    |
            |____|____|____|

                 |----|
                  width

   You will need to decide on a default spatial width.  Optionally, this can
   be a multiple of a scale factor, passed as an argument.  We will only test
   your code by calling it with scale = 1.0.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

  Arguments:
      image    - a grayscale image in the form of a 2D numpy
      xs       - numpy array of shape (N,) containing x-coordinates
      ys       - numpy array of shape (N,) containing y-coordinates
      scale    - scale factor

   Returns:
      feats    - a numpy array of shape (N,K), containing K-dimensional
                 feature descriptors at each of the N input locations
                 (using the default scheme suggested above, K = 72)
"""

def single_box(x, y, theta):
    #helper function to put gradient directions in orientation bins for a 3x3 grid
    histogram = np.zeros(8)
    for i in range(0, 3):
        for j in range(0, 3):
            #depending on the value of theta we place it in a particular bin
            if theta[x+i, y+j] >= -np.pi and theta[x+i, y+j] < -0.75 * np.pi:
                histogram[0] += 1
            elif theta[x+i, y+j] >= -0.75 * np.pi and theta[x+i, y+j] < -0.5 * np.pi:
                histogram[1] += 1
            elif theta[x+i, y+j] >= -0.5 * np.pi and theta[x+i, y+j] < -0.25 * np.pi:
                histogram[2] += 1
            elif theta[x+i, y+j] >= -0.25 * np.pi and theta[x+i, y+j] < 0:
                histogram[3] += 1
            elif theta[x+i, y+j] >= 0 * np.pi and theta[x+i, y+j] < 0.25 * np.pi:
                histogram[4] += 1
            elif theta[x+i, y+j] >= 0.25 * np.pi and theta[x+i, y+j] < 0.5 * np.pi:
                histogram[5] += 1
            elif theta[x+i, y+j] >= 0.5 * np.pi and theta[x+i, y+j] < 0.75 * np.pi:
                histogram[6] += 1
            elif theta[x+i, y+j] >= 0.75 * np.pi and theta[x+i, y+j] <= np.pi:
                histogram[7] += 1
    return(histogram)

def extract_features(image, xs, ys, scale = 1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   N = len(xs)
   feats = []
   #padding the image since we look around its neighbourhood
   padded_image = pad_border(image, 4, 4)
   dx, dy = sobel_gradients(padded_image)
   theta = np.arctan2(dy, dx)
   for i in range(N):
       #finding the interest points
       cx, cy = xs[i], ys[i]
       #subtracting by 4 since we have a 9x9 grid and assume interest point to be in the middle
       start_x = cx - 4
       start_y = cy - 4
       all_hist = []
       for j in range(3):
           for k in range(3):
               #appending the histogram for one 3x3 grid
               block_hist = single_box(start_x + 3 * j, start_y + 3 * k, theta)
               #stacking these histograms together
               all_hist = np.hstack((all_hist, block_hist))
       feats.append(all_hist)
   feats = np.asarray(feats)
   return feats

"""
   FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)

   Given two sets of feature descriptors, extracted from two different images,
   compute the best matching feature in the second set for each feature in the
   first set.

   Matching need not be (and generally will not be) one-to-one or symmetric.
   Calling this function with the order of the feature sets swapped may
   result in different returned correspondences.

   For each match, also return a real-valued score indicating the quality of
   the match.  This score could be based on a distance ratio test, in order
   to quantify distinctiveness of the closest match in relation to the second
   closest match.  It could optionally also incorporate scores of the interest
   points at which the matched features were extracted.  You are free to
   design your own criterion. Note that you are required to implement the naive
   linear NN search. For 'lsh' and 'kdtree' search mode, you could do either to
   get full credits.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices. You are required to report the efficiency comparison
   between different modes by measure the runtime (check the benchmarking related
   codes in hw2_example.py).

   Arguments:
      feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                 feature descriptors (generated via extract_features())
      feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                 feature descriptors (generated via extract_features())
      scores0  - a numpy array of shape (N0,) containing the scores for the
                 interest point locations at which feats0 was extracted
                 (generated via find_interest_point())
      scores1  - a numpy array of shape (N1,) containing the scores for the
                 interest point locations at which feats1 was extracted
                 (generated via find_interest_point())
      mode     - 'naive': performs a brute force NN search

               - 'lsh': Implementing the local senstive hashing (LSH) approach
                  for fast feature matching. In LSH, the high dimensional
                  feature vectors are randomly projected into low dimension
                  space which are further binarized as boolean hashcodes. As we
                  group feature vectors by hashcodes, similar vectors may end up
                  with same 'bucket' with high propabiltiy. So that we can
                  accelerate our nearest neighbour matching through hierarchy
                  searching: first search hashcode and then find best
                  matches within the bucket.
                  Advice for impl.:
                  (1) Construct a LSH class with method like
                  compute_hash_code   (handy subroutine to project feature
                                      vector and binarize)
                  generate_hash_table (constructing hash table for all input
                                      features)
                  search_hash_table   (handy subroutine to search hash table)
                  search_feat_nn      (search nearest neighbour for input
                                       feature vector)
                  (2) It's recommended to use dictionary to maintain hashcode
                  and the associated feature vectors.
                  (3) When there is no matching for queried hashcode, find the
                  nearest hashcode as matching. When there are multiple vectors
                  with same hashcode, find the cloest one based on original
                  feature similarity.
                  (4) To improve the robustness, you can construct multiple hash tables
                  with different random project matrices and find the closest one
                  among all matched queries.
                  (5) It's recommended to fix the random seed by random.seed(0)
                  or np.random.seed(0) to make the matching behave consistenly
                  across each running.

               - 'kdtree': construct a kd-tree which will be searched in a more
                  efficient way. https://en.wikipedia.org/wiki/K-d_tree
                  Advice for impl.:
                  (1) The most important concept is to construct a KDNode. kdtree
                  is represented by its root KDNode and every node represents its
                  subtree.
                  (2) Construct a KDNode class with Variables like data (to
                  store feature points), left (reference to left node), right
                  (reference of right node) index (reference of index at original
                  point sets)and Methods like search_knn.
                  In search_knn function, you may specify a distance function,
                  input two points and returning a distance value. Distance
                  values can be any comparable type.
                  (3) You may need a user-level create function which recursively
                  creates a tree from a set of feature points. You may need specify
                  a axis on which the root-node should split to left sub-tree and
                  right sub-tree.


   Returns:
      matches  - a numpy array of shape (N0,) containing, for each feature
                 in feats0, the index of the best matching feature in feats1
      scores   - a numpy array of shape (N0,) containing a real-valued score
                 for each match
"""

def euclid_distance(a, b):
    #helper function to find the distance between two vectors using euclidean norm
    distance = np.linalg.norm(a-b)
    return distance

def match_features(feats0, feats1, scores0, scores1, mode='naive'):
    matches = np.zeros(len(feats0))
    scores = np.zeros(len(feats0))
    if mode == 'naive':
        for i in range(len(feats0)):
            min = 10000000
            second_min = 1000000
            match = 0
            for j in range(len(feats1)):
                dist = euclid_distance(feats0[i], feats1[j])
                if dist < min:
                    second_min = min
                    min = dist
                    match = j
            matches[i] = match
            scores[i] = min/second_min
    else:
        #kdtree
        tree = build_kdtree(feats1)
        #build the tree of feature vectors in feats1
        for i in range(len(feats0)):
            #searching for the nearest neighbour of feature vector in feats 0
            match_feat = kdtree_searchnn(tree, feats0[i])
            #finding its index in feats1
            ind = np.where(np.all(feats1 == match_feat, axis = 1))[0][0]
            dist1 = euclid_distance(feats0[i], feats1[ind])
            matches[i] = ind
            scores[i] = dist1
    return matches.astype(int), scores

"""
   HOUGH TRANSFORM (7 Points Implementation + 3 Points Write-up)

   Assuming two images of the same scene are related primarily by
   translational motion, use a predicted feature correspondence to
   estimate the overall translation vector t = [tx ty].

   Your implementation should use a Hough transform that tallies votes for
   translation parameters.  Each pair of matched features votes with some
   weight dependant on the confidence of the match; you may want to use your
   estimated scores to determine the weight.

   In order to accumulate votes, you will need to decide how to discretize the
   translation parameter space into bins.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      xs0     - numpy array of shape (N0,) containing x-coordinates of the
                interest points for features in the first image
      ys0     - numpy array of shape (N0,) containing y-coordinates of the
                interest points for features in the first image
      xs1     - numpy array of shape (N1,) containing x-coordinates of the
                interest points for features in the second image
      ys1     - numpy array of shape (N1,) containing y-coordinates of the
                interest points for features in the second image
      matches - a numpy array of shape (N0,) containing, for each feature in
                the first image, the index of the best match in the second
      scores  - a numpy array of shape (N0,) containing a real-valued score
                for each pair of matched features

   Returns:
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
      votes   - a matrix storing vote tallies; this output is provided for
                your own convenience and you are free to design its format
"""
class kdnode():
    #initializing the class with left tree, right tree and feature parameters
    def __init__(self, features = None, left = None, right = None):
        self.left = left
        self.right = right
        self.features = features

def build_kdtree(features, depth = 0):
    k = 72
    n = len(features)
    #base case condition
    if n <= 0:
        return None
    axis = depth % k
    #sorting the feature vectors based on axis
    sorted_feats = sorted(features, key = lambda features:features[axis])
    #recursively putting feature vectors into nodes in the tree
    tree = kdnode(features = sorted_feats[n//2], left = build_kdtree(sorted_feats[:n//2], depth + 1), right = build_kdtree(sorted_feats[n//2 + 1: ], depth + 1))
    return tree

def near_distance(pivot, f1, f2):
    #helper function to find nearest distance to a particular pivot point
    if f1 is None:
        return f2
    elif f2 is None:
        return f1
    if euclid_distance(pivot, f1) < euclid_distance(pivot, f2):
        return f1
    else:
        return f2

def kdtree_searchnn(root, feature, depth = 0, best_feat = None):
    k = len(feature)
    #base case condition
    if root is None:
        return best_feat
    axis = depth % k
    opposite_branch = None
    next_branch = None
    #dpeending on the axis point, we traverse in a particular direction
    if feature[axis] < root.features[axis]:
        next_branch = root.left
        opposite_branch = root.right
    else:
        next_branch = root.right
        opposite_branch = root.left
    #find the nearest distance of input feature to current node and minimum node previously through recursion
    best_feat = near_distance(feature, kdtree_searchnn(next_branch, feature, depth + 1), root.features)
    #traversing the other side of the tree
    if euclid_distance(feature, best_feat) > abs(feature[axis] - root.features[axis]):
        best_feat =  near_distance(feature, kdtree_searchnn(opposite_branch, feature, depth + 1), best_feat)
    return best_feat

def hough_votes(xs0, ys0, xs1, ys1, matches, scores):
   dx = (np.zeros((len(xs0)))).astype(int)
   dy = (np.zeros((len(xs0)))).astype(int)
   #finding the x and y offesets depending on the feature amtches
   for i in range(len(xs0)):
       dx[i] = int(xs0[i] - xs1[matches[i]])
       dy[i] = int(ys0[i] - ys1[matches[i]])
   min_x = np.min(dx)
   min_y = np.min(dy)
   #constructing a votes matrix depending on the range of dx and dy
   width = int(np.max(dx) - np.min(dx))
   height = int(np.max(dy) - np.min(dy))
   votes = np.zeros((width + 1, height + 1))
   #placing dx and dy on the votes matrix and adding is score on the vote tally
   for j in range(len(dx)):
       x = dx[j] - min_x
       y = dy[j] - min_y
       votes[x,y] += scores[j]
   #finding the coordiantes with the highest score
   max_index_tuple = np.unravel_index(np.argmax(votes), votes.shape)
   tx = max_index_tuple[0] + min_x
   ty = max_index_tuple[1] + min_y
   return tx, ty, votes
