from threading import Thread
import zmq
import time
from itertools import count
import signal
import sys
from highway_env.utils import DIFFICULTY_LEVELS
running = True
request_id = count()
context = zmq.Context()
pair_socket = context.socket(zmq.PAIR)
pair_socket.connect("tcp://10.10.10.11:8891")
sub_socket = context.socket(zmq.SUB)
sub_socket.connect("tcp://10.10.10.11:8892")
sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')
def signal_handler(sig=None, frame=None):
    pair_socket.send_json({
        'commands': [
            {
                'command': 'unload_map',
            },
            {
                'command': 'destroy_object',
                'object_id': "*",
            }
        ]})
    print('You pressed Ctrl+C!')
    sub_socket.close()
    pair_socket.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
#print(123)
#init_postion = -15 *  DIFFICULTY_LEVELS['HARD']['vehicles_count'] + 150
init_postion = 50
init = {"command": "load_map", "map": "test123.osm", }
def subscribe(id):
    return {
        'command': 'subscribe',
        'request_id': str(next(request_id)),
        'object_id': str(id),
        'keys': [
            'location',
            'rotation'
        ]
    }
# init
pair_socket.send_json({
                'commands': [init]
            })

class comm():

    def __init__(self,car_count):
        self.messages = {}
        self.control = {
            "command": "drive_object",
            "object_id": str(car_count),
            "throttle": 0,
            "brake": 1,
            "steering": 0
        }
        self.new = [
            {
                "command": "spawn_object",
                "object_id": str(car_count),
                "object_type": "carsim",
                "location": [init_postion, 5, 0],
                "rotation": [0, 0, 0],
                "init_speed": 10
            }
        ]
        for i in range(car_count):
            self.new.append({
                'command': 'spawn_object',
                'object_id': str(i),
                'object_type': 'suv',
                'location': [0,i * 5,0],
                'rotation': [0,0,0]
            })

            self.messages[i] = {
                       'command': 'move_object',
                       'object_id': str(i),
                       "location": [0, 0, 0],
                       "rotation": [0, 0, 0]
                   }
        pair_socket.send_json({'commands': self.new + [self.control]})

        self.request = {
            'command': 'request',
            'request_id': str(next(request_id)),
            'object_id': str(car_count),
            'keys': [
                'location',
                'rotation',
                'global_speed'
            ]
        }
        self.request_other = {
            'command': 'request',
            'request_id': str(next(request_id)),
            'object_id': str(0),
            'keys': [
                'location',
                'rotation',
                'global_speed'
            ]
        }
    def send_message(self,messages):
        for k,v in messages.items():
            self.messages[k]['location'] = v['location']
            self.messages[k]['rotation'] = v['rotation']
        s = list(self.messages.values())
        pair_socket.send_json({'commands': s})


    def send_request_other(self,):
        pair_socket.send_json({'commands': [self.request_other]})
        return sub_socket.recv_json()['replies'][0]

    def move_carsim(self,json):
        pair_socket.send_json({'commands': [json]})
        return sub_socket.recv_json()
    def send_control(self,control):
        for k in ["throttle", "brake", "steering"]:
            self.control[k] = control[k]
        pair_socket.send_json({'commands': [self.control]})
        pair_socket.send_json({'commands': [self.request]})
        #pair_socket.send_json({'commands': [self.request_other]})
        return_data = sub_socket.recv_json()['replies'][0]
        #print(control)
        if 'error' in return_data:
            return {
                "location": [init_postion, 5, 0],
                "rotation": [0, 0, 0]
            }
        return return_data['reply']
        #return pair_socket.send_json({'commands': [self.request]})['responses'][0]['reply']

#m = comm(DIFFICULTY_LEVELS['HARD']['vehicles_count'])

if __name__ == '__main__':
    import yaml
    c = yaml.load(open('./test_lane.yaml'))
    print(c['StraightLane']['origin'])
    # m = comm(3)
    # #m.test()
    # x = -100
    # time.sleep(1)
    # try:
    #     while True:
    #         # r = m.send_message({0:{'location':[x,-4,0],'rotation':[0,0,0]},
    #         #         1:{'location':[x,0,0],'rotation':[0,0,0]},
    #         #         2:{'location':[x,4,0],'rotation':[0,0,0]}
    #         #         })
    #         #print(r)
    #         r = m.send_control({
    #            "throttle": 0.,
    #            "brake": 0.,
    #            "steering": 0.
    #         })
    #         print(r)
    #         #print(r['subscriptions'][0]['reply']['location'])
    #         #print(r['subscriptions'])
    #
    #         x+=0.1
    #         time.sleep(0.01)
    # except Exception as e:
    #     print(e)
    #     signal_handler()
