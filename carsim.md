## 功能列表
- 地图建模
    - OSM(需经过转换，建模较为简陋，但是可以快速从高精度地图转换)
    - CityEngine(较为精细，带丰富的道路元素，但需要从高精度地图人工制作)
- 车辆控制
    - 简单位置控制
    - 动力学控制(carsim/physx)
- 动态元素
    *车辆以外的动态元素只能使用简单位置控制*
    
    - 车辆
    - 行人
    - 自行车
    - 摩托车
- 控制
    - 通过zmq异步通讯，来发送指令。如载入地图，生成小车，控制小车，订阅小车数据等。

## 如何开发MSim
*注意仅使用MSim仿真仅需要通过网络接口通讯，不需要执行下列步骤*
1. 安装下列软件
    - Visual Studio 2017 （选中GAMEDEVELOPMENT WITH C++/WIN8.1SDK/UNREAL INSTALLER）
    - Unreal Engine 4.19
    - Unreal Studio Beta  
      1. 打开Epic Game Launcher
      2. 选中左侧Library
      3. 找到UNREAL STUDIO BETA: Unreal Datasmith
      4. 选择Install to Engine 4.19
      
2. 在UE Market中安装插件Carsim和Substance
3. 下载车辆素材 https://cloud.momenta.works/s/qWiI463JiyN530v ，并解压到Content目录下
4. 右击MSim.uproject，选择Generate Visual Studio Project Files
5. 删除部分编译警告。（UE过分严格的错误检查导致部分第三方库无法编译）

    打开`{INSTALL_PATH}\UE_4.19\Engine\Source\Runtime\Core\Public\Windows\WindowsPlatformCompilerSetup.h` 删除27行的`4582` `4583`这两个数字

## 如何使用MSim
*打包发布的版本，非本git项目*
- Carsim 2018.0/2018.1需要在后台运行
  - 保证D:\carsim_data目录下含有carsim的数据文件
  - 保证carsim标题栏显示的有`{ unreal } mkz`字样
- 如果需要自定义的地图，需要存放在`WindowsNoEditor\MSim\Content\Osms`下
- **初次使用，建议阅读Example目录下的示例代码**

## 设置
### 步骤
1. 运行MSim.exe
2. 在同一网络内，通过网络接口控制仿真器的运行

### 控制
- 仿真过程中，按`Tab`键可以切换不同的车辆
- 任意车辆可以通过wasd键盘控制，Carsim的车辆由于始终由外部程序控制，暂时不支持键盘控制。（如有需要，请提issue）
- 大部分车辆均为车后视角，Carsim车则额外提供了俯视视角，按`Tab`切换到对应车辆后按`1`即可切换俯视视角

## 网络接口
MSim仿真系统的全部控制和反馈都通过zmq网络接口发送接收json数据包来进行

示例代码在Example目录下

| |地址|模式|
|--|--|--|
|发送命令|tcp://server_ip:8891|zmq.PAIR|
|接收命令|tcp://server_ip:8892|zmq.SUB|

### 发送请求
- 一个json的数据包可以包含多个Command，command格式见下文

    ``` json
    {
        "commands": [
            {
                "command": "...",
                "...": "..."
            },
            {
                "command": "...",
                "...": "..."
            }
        ]
    }
    ```

- 错误返回

    ```json
    {
        "error": "invalid command",
        "command": "...",
        "...": "..."
    }


### 支持的请求
#### load_map
- map: 带车道线、锚点等信息的msim专用osm地图文件
- example

    ``` json
    {
        "command": "load_map",
        "map": "your_map.osm",
    }
    ```

#### unload_map
- example

    ``` json
    {
        "command": "unload_map",
    }
    ```

#### spawn_object
- 生成car/bicycle/pedestrian/…
- object_id:对象的唯一编号
- object_type: 对象类型
    - carsim(动力学控制)
    - suv/hatchback/sedane(动力学或坐标控制)
    - pedestrian(坐标控制)
    - motorcycle(坐标控制)
    - bicycle(坐标控制)
- init_speed: 初速度，当对象类型为carsim时，可以设定初速度
- example

    ```json
    {
        "command": "spawn_object",
        "object_id": "{object_id}",
        "object_type": "carsim",
        "location": [0, 0, 0],
        "rotation": [0, 0, 0],
        "init_speed": 19.4
    }
    ```

#### destroy_object
- example

    ```json
    {
        "command": "destroy_object",
        "object_id": "{object_id}/*"
    }
    ```

#### set_object
- 设置额外对象参数
- 设置感知区域 perception_length(m), perception_fov(degree)
- [carsim]强制设置当前速度（将重置carsim），暂时不保证可用
- example

	```json
	{
		"command": "set_object",
		"object_id": "{object_id}",
		"init_speed": 15,
		"perception_length": 100,
		"perception_fov": 60
	}
	```

#### move_object
- 移动（传送）对象。当移动carsim车辆时，不保证过程中取得的参数符合动力学。
- example

    ```json
    {
        "command": "move_object",
        "object_id": "{object_id}",
        "location": [0, 0, 0],
        "rotation": [0, 0, 0]
    }
    ```

#### set_object_path
- 设置对象轨迹
- path: 数组中每个元素是一个关键帧。每个关键帧包含坐标，方向和时刻。
- path中第一组值的location, rotation设为null，即使用当前位置。
- example

    ```json
    {
        "command": "set_object_path",
        "object_id": "{object_id}",
        "path": [
            {"location": null, "rotation": null, "time": 0.0},
            {"location": [200, 0, 100], "rotation": [0, 0, 0], "time": 5.0},
            {"location": [300, 0, 100], "rotation": [0, 0, 0], "time": 10.0},
            {"location": [400, 0, 100], "rotation": [0, 0, 0], "time": 15.0}
        ]
    }
    ```

#### drive_object
- 动力学控制对象
- example

    ```json
    {
        "command": "drive_object",
        "object_id": "{object_id}",
        "throttle": 0.5,
        "brake": 0,
        "steering": -0.3
    }
    ```

#### request_object
- 请求对象的信息
- 支持的key
  - location
  - rotation, （roll, pitch, yaw）角度描述的车头方向  
  - heading
  - bounding_box
  - global_speed 全局坐标系下的速度
    - object_type=carsim, 物理引擎中计算得到的速度
    - object_type=others, 按帧差分得到的速度
  - lane_lines 车道线感知，返回感知区域内的车道线2D点集。默认感知距离60m，fov为60°。
  - Carsim列表中的任意参数，见https://cloud.momenta.works/s/pSnDgJX1P6DYKj0
- example-request

    ```json
    {
        "command": "request",
        "request_id": "{request_id}",
        "object_id": "first_car",
        "keys": ["location", "rotation", "heading", "bounding_box", "global_speed", "lane_lines", "OTHER_CARSIM_PARAMS"]
    }
    ```

- example-reply

    ```json
    {
        "responses": [
            {
                "request_id": "{request_id}",
                "reply": {
                    "location": [0, 0, 0],
                    "rotation": [0, 0, 90],
                    "bounding_box": []
                }
            }
        ]
    }
    ```
    
#### subscribe
- 订阅对象的信息，成功后每一帧都会返回信息
- 支持的key与request一致
- subscribe中的object_id可以为*，即对所有对象作相同的订阅
- example-request

    ```json
    {
        "command": "subscribe",
        "request_id": "1",
        "object_id": "{object_id}/*",
        "keys": ["location", "rotation", "heading", "bounding_box", "OTHER_CARSIM_PARAMS"]
    }
    ```

#### unsubscribe
- 按request_id取消订阅，request_id可以为*，即取消所有对象的订阅
- example
	
	```json
	{
		"command": "unsubscribe",
		"request_id": "{request_id}/*"
	}
	```
	
#### request_traffic_light
- 返回场景中所有红绿灯的位置和编号
- example

    ```json
    {
        "command": "request_traffic_light",
        "request_id": "1"
    }
    ```
    
#### set_traffic_light
- 根据得到的红绿灯编号，设置红绿灯状态
- id可以为*，即设置所有红绿灯状态
- example

    ```json
        {
            "command": "set_traffic_light",
            "settings": [
                {"id": "TrafficLights5_2", "status": "green"}
            ]
        }
    ```

## 其他
- Location

    右手正交坐标系，其中车头的起始方向为(1, 0, 0)

- Rotation

    roll, pitch, yaw分别对应(x, y, z)轴，以右手定则为旋转正方向, 0° ~ 360°

- Throttle

    0.0 ~ 1.0
- Brake

    0.0 ~ 1.0
- Steering

    -1.0 ~ 1.0
- 仿真时序

    仿真器的每一帧进行一次物理计算，帧率通常在30-60帧之间。因此每帧时间在16毫秒到33毫秒之间。