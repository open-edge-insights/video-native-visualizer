# Intel Edge Insights Simple Visualizer
Simple visualizer for the EIS platform.

### 1. Running as a docker container

#### Steps to build and run viualizer

* Follow [provision/README.md](../README#provision-eis.md) for EIS provisioning
  if not done already as part of EIS stack setup

* Running visualizer as a container from [docker_setup](../../docker_setup):

  ```
    $ docker-compose up --build ia_visualizer
  ```

-----
**NOTE**:
1. The admin has to make sure all the necessary config is set in etcd before starting the visualizer.
2. The user has to make sure the path provided in docker-compose volumes of visualizer correlates to the one in etcd before running visualizer if he wishes to save images.
3. Run this command in terminal if you run into tkinter couldn't connect to display exception

   ```sh
   $ xhost +
   ```
-----

* If one needs to remove the classified images on a periodic basis:

  1. Have this command running in a separate terminal as a cleanup task to remove images older than 60 mins in IMAGE_DIR. Replace <path-to-IMAGE_DIR> with IMAGE_DIR path given while running visualizer. The -mmin option can be changed accordingly by the user.

    ```sh
    $ while true; do find <path-to-IMAGE_DIR> -mmin +60 -type f -name "*.png" -exec rm -f {} \;;  done
    ```

#### Using Labels

  In order to have the visualizer label each of the defects on the image, labels in JSON format has to be provided in [etcd_pre_load.json](../docker_setup/provision/config/etcd_pre_load.json) file under "/Visualizer/config" with the mapping between the topic subscribed and the text that has to be displayed.

  An example of what this JSON value should look like is shown below. In this case
  it is assumed that the classification types are `0` and `1` and the text labels
  to be displayed are `MISSING` and `SHORT` respectively.

  ```json
  {
      "0": "MISSING",
      "1": "SHORT"
  }
  ```
  > **NOTE:** These labels are the mapping for the PCB demo provided in EIS's visualizer directory. Currently camera1_stream_results consists of pcb demo labeling and camera2_stream_results consists of safety demo labeling.
  Hence, in [etcd_pre_load.json](../docker_setup/provision/config/etcd_pre_load.json) proper mapping of camera1_stream_results, camera2_stream_results (subscribed topics) should be done with pcb demo labeling, safety demo labeling respectively.

```json
  "/Visualizer/config": {
      "display": "true",
      "save_image": "false",
      "labels" : {
          "camera1_stream_results": {
              "0": "MISSING",
              "1": "SHORT"
          },
          "camera2_stream_results":{
              "1": "safety_helmet",
              "2": "safety_jacket",
              "3": "Safe",
              "4": "Violation"
          }
      }
  }
```