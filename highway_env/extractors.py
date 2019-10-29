import numpy as np
import math

np.set_printoptions(suppress=True, precision=4)

# import Road
"""
tips:

see details at 'Class Extractor' about how to use.
Based on solo agent.
Will update the extractors if multi-agent is necessary     
     
"""


class VecE2(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dtype = 'VecE2'

    def __add__(self, vec):
        tx = self.x + vec.x
        ty = self.y + vec.y
        return VecE2(tx, ty)

    def __sub__(self, vec):
        tx = self.x - vec.x
        ty = self.y - vec.y
        return VecE2(tx, ty)


class VecSE2(object):
    def __init__(self, x, y, thet):
        self.x = x
        self.y = y
        self.thet = thet


class VecSE2_and_v(object):
    def __init__(self, x, y, thet, speed):
        self.x = x
        self.y = y
        self.thet = thet
        self.speed = speed


class LineSegment(object):
    def __init__(self, a, b):
        assert (a.dtype == 'VecE2')
        assert (b.dtype == 'VecE2')
        self.A = a
        self.B = b


def polar(r, thet):
    return VecE2(r * math.cos(thet), r * math.sin(thet))


def get_signed_area(pts):
    # https://en.wikipedia.org/wiki/Shoelace_formula
    # sign of -1 means clockwise, sign of 1 means counterclockwise
    # from /AutomotiveDrivingModels/src/2d/utils/minkowski.jl
    npts = len(pts)
    retval = pts[npts - 1].x * pts[0].y - pts[0].x * pts[npts - 1].y
    for i in range(npts - 1):
        retval += pts[i].x * pts[i + 1].y
        retval -= pts[i + 1].x * pts[i].y

    return retval / 2


def get_edge(pts, i):
    npts = len(pts)
    a = pts[i]
    b = pts[i + 1] if i + 1 < npts else pts[0]
    return LineSegment(a, b)


def cyclic_shift_left(arr, d, n):
    # print("d : " + str(d))
    for i in range(math.gcd(d, n)):
        temp = arr[i]
        j = i
        while True:
            k = j + d
            if k >= n:
                k = k - n
            if k == i:
                break
            # print("index ~~  " + str(j) + ' ' + str(k))
            arr[j] = arr[k]
            j = k
        arr[j] = temp
    return arr


def sign(a):
    if a > 0:
        return 1
    elif a == 0:
        return 0
    else:
        return -1


def ensure_pts_sorted_by_min_polar_angle(poly):
    npts = len(poly)
    assert (npts >= 3)
    assert (sign(get_signed_area(poly)) == 1)

    angle_start = float('inf')
    index_start = -1
    for i in range(npts):
        seg = get_edge(poly, i)

        thet = math.atan2(seg.B.y - seg.A.y, seg.B.x - seg.A.x)
        # print("cur thet :  " + str(thet))
        if thet < 0.0:
            thet = thet + 2 * math.pi
        if thet < angle_start:
            angle_start = thet
            index_start = i
    if index_start != 0:
        poly = cyclic_shift_left(poly, index_start, npts - 1)
    return poly


def isapprox(a, b, atol):
    if np.abs(a - b) < atol:
        return True
    else:
        return False


def abs2(a):
    return a.x * a.x + a.y * a.y


def are_collinear(a, b, c, tol=1e-8):
    val = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)
    return np.abs(val) < tol


def cross(a, b):
    return a.x * b.y - a.y * b.x


def dot(a, b):
    return a.x * b.x + a.y * b.y


def hypot(a, b):
    return math.sqrt(a * a + b * b)


def proj(a, b):
    return (a.x * b.x + a.y * b.y) / hypot(b.x, b.y)


def get_intersection_time(veh, speed, seg):
    o = VecE2(veh.x, veh.y)
    v1 = o - seg.A
    v2 = seg.B - seg.A
    v3 = polar(1.0, veh.thet + math.pi / 2)
    # npv1 = np.array([v1.x,v1.y])
    # npv2 = np.array([v2.x,v2.y])
    # npv3 = np.array([v3.x,v3.y])
    denom = dot(v2, v3)

    if not isapprox(denom, 0.0, atol=1e-10):
        d1 = cross(v2, v1) / denom
        t2 = dot(v1, v3) / denom
        if 0.0 <= d1 and 0.0 <= t2 and t2 <= 1.0:
            return d1 / speed
    else:
        if are_collinear(VecE2(veh.x, veh.y), seg.A, seg.B):
            # npsegA = np.array([seg.A.x,seg.A.y])
            # npo = np.array([o.x,o.y])
            # npsegB = np.array([seg.B.x,seg.B.y])
            dist_a = abs2(seg.A - o)
            dist_b = abs2(seg.B - o)
            # print("dist_a " + str(dist_a))
            # print("dist_b " + str(dist_b))
            # print(np.min(dist_a,dist_b))
            temp = np.sqrt(np.minimum(dist_a, dist_b)) / speed
            return temp
    return float('inf')


def get_collision_time(ray, poly, ray_speed):
    min_col_time = float('inf')
    for i in range(len(poly)):
        seg = get_edge(poly, i)
        col_time = get_intersection_time(ray, ray_speed, seg)
        if col_time < min_col_time:
            min_col_time = col_time

    return min_col_time


def to_oriented_bounding_box(veh):
    lenth = veh.LENGTH
    width = veh.WIDTH
    # print("veh.LENGTH and WIDTH : " + str(veh.LENGTH) + ' ' + str(veh.WIDTH) + '\n')
    # print("position  heading " + str(veh.position[0]) + ' ' + str(veh.position[1]) + ' '+ str(veh.heading))
    # print(" polar : " + str(polar(veh.position[0],veh.position[1]).x + str(polar(veh.position[0],veh.position[1]).y)))
    center = VecSE2(veh.position[0], veh.position[1], veh.heading)
    assert (lenth > 0)
    assert (width > 0)
    assert (not math.isnan(center.thet))
    assert (not math.isnan(center.x))
    assert (not math.isnan(center.y))
    retval = []
    x = polar(lenth / 2, center.thet)
    y = polar(width / 2, center.thet + math.pi / 2)
    # print('x : ' + str(x.x) + ' ' + str(x.y))
    # print('y : ' + str(y.x) + ' ' + str(y.y))
    C = VecE2(center.x, center.y)
    # print(' C : ' + str(C.x) + ' ' + str(C.y))
    # temp = x - y + C
    # temp = x - y + C
    # temp = x - y + C
    # print(' temp : ' + str(temp.x) + ' ' + str(temp.y))
    retval.append(x - y + C)
    retval.append(x + y + C)
    retval.append(VecE2(0, 0) - x + y + C)
    retval.append(VecE2(0, 0) - x - y + C)
    # print('init_retval 0 : ' + str(retval[0].x)  + '  ' + str(retval[0].y) + '\n')
    # print('init_retval 1 : ' + str(retval[1].x)  + '  ' + str(retval[1].y) + '\n')
    # print('init_retval 2 : ' + str(retval[2].x)  + '  ' + str(retval[2].y) + '\n')
    # print('init_retval 3 : ' + str(retval[3].x)  + '  ' + str(retval[3].y) + '\n')
    retval = ensure_pts_sorted_by_min_polar_angle(retval)
    return retval


def get_lane_width():
    return 4


def get_markerdist_left(t):
    lane_width = get_lane_width()
    return lane_width / 2 - t


def get_markerdist_right(t):
    lane_width = get_lane_width()
    return lane_width / 2 + t


def sinlane_derivative(x, A, B, C):
    return A * B * np.cos(B * x + C)


def sinlane_derivative_derivative(x, A, B, C):
    return -A * B * B * np.sin(B * x + C)


def get_curvature(x, lane):
    amplitude = lane.amplitude
    pulsation = lane.pulsation
    phase = lane.phase
    derivative = sinlane_derivative(x, amplitude, pulsation, phase)
    deri_derivative = sinlane_derivative_derivative(x, amplitude, pulsation, phase)
    result = abs(deri_derivative) / math.pow((1 + derivative * derivative), 1.5)

    return result


def CoreFeatureExtractor(vehicle):
    features = np.zeros(8)
    curvature = 0
    features[0] = round(
        (vehicle.lane.local_coordinates(vehicle.position)[1] + vehicle.WIDTH / 2 - get_lane_width() / 2), 4)
    # features[0] = vehicle.position[1] - (vehicle.lane_index*4 + 2)  # vehicle's y
    vehicle_heading = vehicle.heading
    vehicle_lane = vehicle.lane
    lane_heading = vehicle_lane.heading_at(vehicle_lane.local_coordinates(vehicle.position)[0])
    lane_relative_heading = vehicle_heading - lane_heading
    features[1] = round(lane_relative_heading, 4)  # vehicle's heading (direction along the lane is positive )
    features[2] = round(vehicle.velocity, 4)  # vehicle's velocity .
    features[3] = round(vehicle.LENGTH, 4)
    features[4] = round(vehicle.WIDTH, 4)
    features[5] = get_curvature(vehicle.lane.local_coordinates(vehicle.position)[0], vehicle.lane)  # lane's curvature
    d_ml = get_lane_width() / 2 - features[0]
    d_mr = (vehicle.lane.local_coordinates(vehicle.position)[1] + vehicle.WIDTH / 2)
    features[6] = round(d_ml, 4)
    features[7] = round(d_mr, 4)
    # print(" posF_t : " +  str(features[0]) + '\n')
    # print(" posF_thet : " +  str(features[1]) + '\n')
    # print(" v : " +  str(features[2]) + '\n')
    # print(" length : " +  str(features[3]) + '\n')
    # print(" width : " +  str(features[4]) + '\n')
    # print(" curvature : " +  str(features[5]) + '\n')
    # print(" d_ml : " +  str(features[6]) + '\n')
    # print(" d_mr : " +  str(features[7]) + '\n')
    return features


def TemporalFeatureExtractor(vehicle, dt):
    features = np.zeros(10)
    vehicle_lane = vehicle.lane

    lane_heading = vehicle_lane.heading_at(vehicle_lane.local_coordinates(vehicle.position)[0])
    features[0] = round(vehicle.acc, 4)  # vehicle's acc
    features[1] = round((vehicle.acc - vehicle.pre_acc) / dt, 4)  # vehicle's acc's derivation
    features[2] = round((vehicle.heading - vehicle.pre_heading) / dt, 4)  # vehicle's turn_heading_rate_global
    features[3] = round((vehicle.steering_angle - vehicle.pre_steering_angle) / dt, 4)  # vehicle's steering_rate_global
    features[4] = round((
                                vehicle.heading - vehicle.pre_heading - lane_heading) / dt,
                        4)  # vehicle's turn_heading_rate_LaneRelative
    features[5] = round((
                                vehicle.steering_angle - vehicle.pre_steering_angle - lane_heading) / dt,
                        4)  # vehicle's steering_rate_LaneRelative
    features[6], features[7] = get_timegap_features(vehicle)
    features[8], features[9] = get_inv_ttc_features(vehicle)
    return features


def get_timegap_features(vehicle):
    timegap_censor = 30
    vehicle_v = vehicle.velocity
    v_front, _ = vehicle.road.neighbour_vehicles(vehicle)
    if vehicle_v <= 0 or not v_front:
        features6 = timegap_censor
        features7 = 0.0
    else:
        delta_s = vehicle.lane_distance_to(v_front)
        delta_s = delta_s - 2 * vehicle.LENGTH / 2  # distance: this vehicle's front bumper ---- pre vehicle's back bumper
        if delta_s > 0.0:
            # time gap
            features6 = delta_s / vehicle_v
            features7 = 0.0
        else:
            features6 = 0.0
            features7 = 0.0  # collision!
    # features7 = 0.0
    return round(features6, 4), round(features7, 4)


def get_inv_ttc_features(vehicle):
    invttc_censor = 30
    v_front, _ = vehicle.road.neighbour_vehicles(vehicle)
    if not v_front:
        feature8 = invttc_censor
        feature9 = 1.0  # ----------------------_!!
    else:
        delta_s = vehicle.lane_distance_to(v_front)
        delta_s = delta_s - 2 * vehicle.LENGTH  # distance: this vehicle's front bumper ---- pre vehicle's back bumper
        delta_v = v_front.velocity - vehicle.velocity
        if delta_s < 0.0:  # collision!
            ttc = 1.0 / invttc_censor
            feature8 = ttc
            feature9 = 0.0
        elif delta_v > 0.0:  # front car is pulling away
            feature8 = invttc_censor
            feature9 = 0.0
        else:
            f = -delta_v / delta_s
            if f > invttc_censor:
                ttc = 1.0 / f
                if ttc > invttc_censor:
                    feature8 = invttc_censor
                    feature9 = 0.0
                else:
                    feature8 = ttc
                    feature9 = 0.0
            else:
                if f == 0.0:
                    feature8 = invttc_censor
                    feature9 = 0.0
                else:
                    ttc = 1.0 / f
                    if ttc > invttc_censor:
                        feature8 = invttc_censor
                        feature9 = 0.0
                    else:
                        feature8 = ttc
                        feature9 = 0.0
    return round(feature8, 4), round(feature9, 4)


def collision(veh):
    for other in veh.road.vehicles:
        if veh.check_collision(other):
            return True
    return False


def off_lane(d_ml, d_mr, veh_width):
    if (d_ml - veh_width / 2) > 0 or (d_mr - veh_width / 2) > 0:
        return False
    else:
        return True


def negative_velocity(vehicle):
    if vehicle.velocity < 0:
        return True
    else:
        return False


def get_RoadEdgeDistLeft_features(vehicle, d_ml):
    # lane_lateral_offset = vehicle.lane.local_coordinates(vehicle.position)[1]
    lane_index = vehicle.lane_index
    count_lanes = lane_index[2] - 0
    return count_lanes * 4 + d_ml


def get_RoadEdgeDistRight_features(vehicle, d_mr):
    # lane_lateral_offset = get_lane_width() - vehicle.lane.local_coordinates(vehicle.position)[1]
    lane_index = vehicle.lane_index
    # all_side_lanes = vehicle.road.network.all_side_lanes(lane_index)
    all_side_lanes = len(vehicle.road.network.graph[lane_index[0]][lane_index[1]])
    # print("all_side_lanes:", all_side_lanes)
    # biggest_lane = all_side_lanes[-1][-1]
    return 4 * (all_side_lanes - 1 - lane_index[2]) + d_mr


def WellBehavedFeatureExtractor(vehicle):
    features = np.zeros(5)
    local_lateral_offset = get_lane_width() - (vehicle.lane.local_coordinates(vehicle.position)[1] + vehicle.WIDTH / 2)
    d_ml = local_lateral_offset
    d_mr = (vehicle.lane.local_coordinates(vehicle.position)[1] + vehicle.WIDTH / 2)
    features[0] = round(collision(vehicle), 2)
    features[1] = round(off_lane(d_ml, d_mr, vehicle.WIDTH), 2)
    features[2] = round(negative_velocity(vehicle), 2)
    features[3] = round(get_RoadEdgeDistLeft_features(vehicle, d_ml), 4)
    features[4] = round(get_RoadEdgeDistRight_features(vehicle, d_mr), 4)
    return features


def ForeForeFeatureExtractor(vehicle):
    features = np.zeros(3)
    distance_censor = 100
    ego_vel = vehicle.velocity
    lane_index = vehicle.lane_index
    no_fore_fore = False
    v_front, _ = vehicle.road.neighbour_vehicles(vehicle, lane_index)
    v_front_front = None
    if v_front:
        v_front_front, _ = vehicle.road.neighbour_vehicles(v_front)
        if v_front_front is None:
            no_fore_fore = True
    else:
        no_fore_fore = True

    if not no_fore_fore:
        distance = vehicle.lane_distance_to(v_front_front)
        # print("test dis:",distance)
        velocity = v_front_front.velocity - ego_vel
        features[0] = round(distance, 4)  # distance from ego-vehicle to fore-fore-car
        features[1] = round(velocity, 4)  # relative velocity based on ego-vehicle
        features[2] = round(v_front_front.acc, 4)  # absolute acc of fore-fore-car
    else:
        features[0] = round(distance_censor, 4)
        features[1] = 0.0
        features[2] = 0.0
    return features


def two_vehicle_globaldist(veh, veh_ego):
    vec1 = np.array((veh.position[0], veh.position[1]))
    vec2 = np.array((veh_ego.position[0], veh_ego.position[1]))
    return np.linalg.norm(vec1 - vec2)


def observe(vehicles, ego_id):
    state = vehicles[ego_id]
    # egoid = id
    state_x = state.position[0]
    state_y = state.position[1]
    state_v = state.velocity
    state_thet = state.heading
    ego_vel = polar(state_v, state_thet)
    nbeams = 20
    lidar_max_range = 100.0
    lidar_angles = np.linspace(0, 2 * math.pi, nbeams + 1)
    i = 0
    ranges = np.zeros(20)
    range_rates = np.zeros(20)
    ranges_id = [-1] * 20
    ranges_id = np.asarray(ranges_id)
    for angle in lidar_angles:

        if i == nbeams:
            break
        ray_angle = state_thet + angle
        ray_vec = polar(1.0, ray_angle)
        ray = VecSE2(state_x, state_y, ray_angle)
        _range = lidar_max_range
        range_rate = 0.0
        range_id = -1
        for veh in vehicles:
            distance = two_vehicle_globaldist(veh, vehicles[ego_id])
            if veh.id != ego_id and distance <= 50:
                poly = to_oriented_bounding_box(veh)
                # print("poly :  \n")
                # print(str(poly[0].x) + "  " + str(poly[0].y) + '\n')
                # print(str(poly[1].x) + "  " + str(poly[1].y) + '\n')
                # print(str(poly[2].x) + "  " + str(poly[2].y) + '\n')
                # print(str(poly[3].x) + "  " + str(poly[3].y) + '\n')
                # print("veh position : " + str(veh.position[0]) + ' ' +  str(veh.position[1]) + '\n')
                # print('\n')
                range2 = get_collision_time(ray, poly, 1.0)
                if not math.isnan(range2) and range2 < _range:
                    _range = range2
                    range_id = veh.id
                    relative_speed = polar(veh.velocity, veh.heading) - ego_vel
                    range_rate = proj(relative_speed, ray_vec)
        ranges[i] = _range
        range_rates[i] = range_rate
        ranges_id[i] = range_id
        i = i + 1
    return ranges, range_rates, ranges_id


def CarLidarFeatureExtractor(vehicles, _id):
    ranges, range_rates, ranges_id = observe(vehicles, _id)
    # print("ranges size : " + str(ranges.shape) + '\n')
    # for i in ranges:
    # print(' ' + str(i) + " ")
    # print("range_rates size : " + str(range_rates.shape) + '\n')
    # for i in range_rates:
    # print(' ' + str(i) + " ")
    # print("ranges_id : " + str(ranges_id.shape) + '\n')
    # for i in ranges_id:
    #     print(' ' + str(i) + " ")
    return ranges, range_rates, ranges_id


class Extractor(object):
    def FeatureExtractor(self, vehicles, ego_id, dt):
        """
        :param： vehicles: all the vehicles in this env(self.road.vehicles)
        :param： ego_id: ego_vehicle's id (eg: 0)
        :param： dt : the simulation time[s] (eg: 1.0)
        :return: 66-dimension features in dt[s] of the ego_vehicle
        """
        # print("test len(vehicle):",len(vehicles))
        features = np.zeros(66)
        corefeatures = CoreFeatureExtractor(vehicles[ego_id])
        temporalfeatures = TemporalFeatureExtractor(vehicles[ego_id], dt)
        behavior = WellBehavedFeatureExtractor(vehicles[ego_id])
        ranges, range_rates,ranges_id = CarLidarFeatureExtractor(vehicles, ego_id)
        foreforefeatures = ForeForeFeatureExtractor(vehicles[ego_id])
        features[0:8] = corefeatures
        features[8:18] = temporalfeatures
        features[18:23] = behavior
        features[23:43] = ranges
        features[43:63] = range_rates
        features[63:66] = foreforefeatures
        # features[66:86] = ranges_id
        # print("features:\n",features)
        return features
