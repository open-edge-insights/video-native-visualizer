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

  In order to have the visualizer label each of the defects on the image (i.e.
  text underneath of the bounding box), you will need to provide a JSON file with
  the mapping between the classfication type and the text you wish to display.

  An example of what this JSON file should look like is shown below. In this case
  it is assumed that the classification types are `0` and `1` and the text labels
  to be displayed are `MISSING` and `SHORT` respectively.

  ```json
  {
      "0": "MISSING",
      "1": "SHORT"
  }
  ```
  > **NOTE:** These labels are the mapping for the PCB demo provided in EIS's visualizer directory. Currently pcb_demo_label.json and safety_demo_label.json files are provided for reference.

  An important thing to note above, is that the keys need to still be strings.
  The visualizer will take care of the conversion when it receives publications
  for classification results.

  In case the user running visualizer as a docker container, the visualizer section in [docker-compose.yml](../docker_setup/docker-compose.yml) file should be changed in order to process the labels from a specific JSON file. The ***command*** variable in docker-compose.yml file can be changed as below for using safety_demo_label.json instead of default json file:
  

  Before
  ```json
  ia_visualizer:
  depends_on:
    - ia_common
  -----snip-----
  command: ["pcb_demo_label.json"]
  -----snip-----

  ```
  After
  ```json
  ia_visualizer:
  depends_on:
  - ia_common
  -----snip-----
  command: ["safety_demo_label.json"]
  -----snip-----
  ```

Passing this json file as command line option has been taken care in corrsponding Docker file.