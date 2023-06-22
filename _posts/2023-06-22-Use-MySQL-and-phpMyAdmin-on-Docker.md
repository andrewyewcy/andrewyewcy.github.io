![mysql_docker_myphpadmin.png](/assets/images/mysql_docker_myphpadmin.png){:class="img-responsive"}

[Previously](https://andrewyewcy.com/Systematically-Web-Scrape-Multiple-Data-Files-from-Websites/), the bicycle rides data from Bixi was web-sraped and stored locally. In this article, we explore how to create a relational database using MySQL, phpMyAdmin, and Docker containers to store the data without installing MySQL, phpMyAdmin, and their dependencies locally.

# Motivation and Introduction

## Why store the data in a database?

**Consolidation**
- The web-scraped data is stored across multiple Comma Separated Value (CSV) files, which makes it more difficult to access all the data since the data files must be loaded individually from each CSV and combined before use.


**Tracking**
- With change being the only constant, no data is expected to be perfect or clean. Using a database with a database management system (DBMS) like MySQL allows for logging and tracking of updates to data.


**Structure, Relationships and Normalization**
- On top of being a DBMS, MySQL is also a relational DBMS, meaning that different data tables can be connected through specified relationships between columns. For example, a bicycle docking station can store many bicycles. Furthermore, large data tables (many columns) can be broken down into smaller tables which are connected through relationships. This is known as normalization and the benefits include smaller data tables and data completeness.


**Data Authority, Sharing and Security**
- Rather than sending the data files through email, other users can access the same consolidated data stored in a database. This reduces the risks of having multiple copies (truths) of data among users, which causes confusion and may lead to wasted time or even wrongly justified decision making. Finally, MySQL also provides a way to manage access levels to data among users.

## What are the tools and why?

**[MySQL](https://www.mysql.com/)**
- MySQL is an open-source relational database management system.  Although not the latest, it is one of the world's most [popular](https://hub.docker.com/_/mysql) DBMS.

**[phpMyAdmin](https://www.phpmyadmin.net/)**
- Without a user interface, MySQL can only be interacted with through Terminal (or PowerShell and Command Prompt depending on OS). phpMyAdmin addresses this problem by being an open-source portable web application that acts as an administration tool for MySQL. An alternative would be [MySQL Workbench](https://www.mysql.com/products/workbench/), but that requires installation on local machine and defeats the purpose of using Docker.

**[Docker](https://www.docker.com/)**
- Docker is a tool that packages software into containers independent of the local system. Docker was used to run MySQL and phpMyAdmin on the local computer without installation, and can be used to package the entire setup in the future. Read here for more details on [Docker](https://www.docker.com/resources/what-container/#:~:text=A%20Docker%20container%20image%20is,tools%2C%20system%20libraries%20and%20settings.)

## Wait, isn't Docker for Software Development?

The ability to containerize applications provided by Docker is useful in the field of data science in the following manner:

**Ease of Installation**
- Using Docker containers avoids the need to install MySQL and all its dependencies on your local computer. Instead of spending hours installing and configuring MySQL to work with phpMyAdmin over many computers, using Docker containers allows users to get MySQL running with phpMyAdmin in less than 5 minutes consistently across many computers.
- Docker is also useful when setting up specific environments for machine learning to be shared among colleagues, and is even more so when bringing the trained models to production in the cloud.

**Sharing and Scaling**
- In the world of big data, the data size is usually too large to store in any single computer. Docker containers allow users to transition their containerized DBMS and volumes onto the cloud, skipping the hassle of reconfiguring setup to match cloud specifications.

**Operations and Modularization**
- Containerized applications can be easily updated or replaced with another container if the current one fails. This means that the entire operations process of DBMS deployment can be modularized into independent components. Individual components (containers) can be replaced without the need to rebuild all operations, reducing database downtime.

# Overview of Setup

![docker_network_setup.png](/assets/images/docker_network_setup.png){:class="img-responsive"}

The image above represents an overview of the DBMS setup. The containers with red caps represent Docker containers running MySQL and phpMyAdmin Docker images. Both the MySQL and phpMyAdmin containers can communicate with each other because they exist in a Docker network, represented by the cloud. The actual data is stored outside the cloud in a data warehouse, which in this case is just the local computer. Note that in the data warehouse, data is stored as Docker volumes, which can be pushed and scaled on cloud services like AWS. The cleaned raw data is ingested through MySQL and stored in the data warehouse. Finally, dashboards and machine learning can access the data through MySQL in the Docker network.

## Requirements

To run the above setup, you will need:
- Docker [installed](https://docs.docker.com/get-docker/) on your local computer
- a web browser, Google Chrome was used in this article
- (optional) a code editor for writing Docker-compose script, [VSCodium](https://vscodium.com/) was used

## Notes on Image Version

For this example, MySQL was pulled using the version 8.0 tag on [Docker Hub](https://hub.docker.com/_/mysql), a public repository for Docker images, which are used to build Docker containers. Similarly, phpMyAdmin was also pulled from [Docker Hub](https://hub.docker.com/_/phpmyadmin), with version 5.2 specified.

Note, Docker images pulled from Docker Hub become Docker containers when they are run on Docker Engine, which runs on the local machine. (Read more [here](https://www.docker.com/resources/what-container/)) What this implies practically is that Docker containers are still dependent on the CPU of the local host, meaning that some Docker images which were built to run on Intel chips may not work on the newer ARM chips like Apple's M1 and M2. However, many Docker images are being or have already been updated to run on ARM chips at the time of this article. This is usually specified on the Docker Hub page of the desired image.

# Deployment of DBMS Setup using Docker Compose

The first step is to create a docker-compose script that tells Docker what containers to setup. 
Once defined, Docker-compose automatically handles:
- the pulling of Docker images from Docker Hub
- the creation of a Docker network
- the creation of Docker containers from Docker images
- the connection between Docker containers within the network
- the connection between the Docker containers to volumes outside the network. 

The Docker-compose script is written as a [YAML](https://en.wikipedia.org/wiki/YAML) file, which is a human readable data serialization language. A GitHub repo for the script can be found [here](https://github.com/andrewyewcy/docker/blob/main/mysql.yaml)

The script is included below with explanations for each line of code.


```console
# Specify version of docker-compose for backwards compatability
version: '3.8'

# Define services (docker containers) to run
services:
  
  # Define a container named mysql to run MySQL DBMS
  mysql:

    # Specify image tag to reference, refer to Docker Hub https://hub.docker.com/_/mysql
    image: mysql:8.0

    # Tells Docker to restart container if fail
    restart: always

    # Define environment variables, different for each image
    environment:
      # rootroot was used in this case as an example
      MYSQL_ROOT_PASSWORD: rootroot

    # Define ports for communication between host (left) and container(right)
    ports:
      - '3306:3306'

    # Define volumne to write data to for data persistence when container restarts
    # Local host directory (left) : directory in container (right)
    volumes:
      - mysql_db_data:/var/lib/mysql

  # Define a container to run phpMyAdmin
  phpmyadmin:

    # Specify phpmyadmin to run after mysql has started
    depends_on:
      - mysql

    # Specify image tag to reference, refer to Docker Hub https://hub.docker.com/_/phpmyadmin
    image: phpmyadmin:5.2

    # Tells Docker to restart container if fail
    restart: always

    # Define ports for communication between host (left) and container(right)
    ports:
      - '8080:80'

    # Define link from mysql container to phpmyadmin container
    links:
      - mysql:db

# Define Docker volumes to store data
volumes:

  # Specify volume name
  mysql_db_data:

    # Tells docker that volume is stored on local computer
    driver: local
```

Now that the Docker-compose script is defined, the beauty of using Docker is that the entire MySQL and phpMyAdmin setup can be executed in 1 line of code as shown below:


```console
# Run below line in terminal, make sure you are in same directory as YAML file
docker-compose -f mysql.yaml up
```

![Screenshot%202023-06-21%20at%2012.21.38.png](/assets/images/Screenshot%202023-06-21%20at%2012.21.38.png){:class="img-responsive"}

Once the docker network and containers are up and running, type the following path into a web browser of your choice to access phpMyAdmin.


```console
# Note, the port 8080 was specified in the YAML file and can be changed accordingly in case of port conflict
http://localhost:8080/
```

![Screenshot%202023-06-21%20at%2012.25.10.png](/assets/images/Screenshot%202023-06-21%20at%2012.25.10.png){:class="img-responsive"}

Within phpMyAdmin, the usual database operations like creating tables and defining primary keys can be performed.

Finally, when done, the entire setup can be shut down using the `down` command, which automatically shuts down and removes the Docker containers and networks. The databases and data are stored in the specified Docker volume in the YAML file, and is persistent when the Docker-compose YAML file is initiated again.


```console
# Run below line in terminal, make sure you are in same directory as YAML file
docker-compose -f mysql.yaml down
```

# Conclusion and Next Steps

In this article, we explored how Docker containers can be used to conveniently setup MySQL and phpMyAdmin on a local computer without the need to deal with installation and dependencies. For now, the data is stored on the local computer. But, in the future when more data is acquired, the entire Docker setup can be packaged and hosted on cloud services like AWS, enabling scalability. In future articles, the Bixi rides data is loaded into the setup database for further cleaning and use in analysis, visualizations and machine learning models.

# Appendix

Here we will explore the step by step progression of building the docker-compose YAML file starting from basic commands.

## A01 Install Docker and Ensure it is running

Docker can be installed using instructions [here](https://docs.docker.com/get-docker/).

Then, Docker can be verified to be running through below methods:
1) Visible Desktop icon of Docker:

![Screenshot%202023-06-19%20at%205.40.22%20PM.png](/assets/images/Screenshot%202023-06-19%20at%205.40.22%20PM.png){:class="img-responsive"}

2) Using terminal:


```console
# Input below to verify both Docker client and server(Engine) are running
docker version
```

![Screenshot%202023-06-19%20at%206.01.50%20PM.png](/assets/images/Screenshot%202023-06-19%20at%206.01.50%20PM.png){:class="img-responsive"}

## A02 Pull Images from Docker Hub

A benefit of Docker is that the large community base and companies have already constructed and uploaded ready-to-use Docker images on Docker Hub. Each image may contain instructions specific to the image. The images used in this article are:
- [MySQL](https://hub.docker.com/_/mysql), version 8.0 was used
- [phpMyAdmin](https://hub.docker.com/_/phpmyadmin), version 5.2 was used

The images can be pulled using the Docker `pull` command:


```console
# If unspecified, the latest version of the image will be pulled
# Specify version after the ':' symbol
docker pull mysql:8.0
docker pull phpmyadmin:5.2
```

![Screenshot%202023-06-19%20at%206.04.08%20PM.png](/assets/images/Screenshot%202023-06-19%20at%206.04.08%20PM.png){:class="img-responsive"}

To check if images are pulled, the `ls` command can be chained with the `image` command:


```console
# Check pulled (downloaded) images
docker image ls
```

![Screenshot%202023-06-19%20at%206.05.21%20PM.png](/assets/images/Screenshot%202023-06-19%20at%206.05.21%20PM.png){:class="img-responsive"}

## A03 Creating a Network to run Containers

A common network is needed to enable the MySQL and phpMyAdmin containers to communicate effectively with each other. To this end, the `create` command can be chained with the `network` command:


```console
# input the name of the network after `create`
docker network create mysql_network

# similar to images, use `ls` to check created networks
docker network ls
```

![Screenshot%202023-06-19%20at%206.13.46%20PM.png](/assets/images/Screenshot%202023-06-19%20at%206.13.46%20PM.png){:class="img-responsive"}

Along with the created network `mysql_network`, the other 3 networks are default Docker networks that should be left running.

## A04 Create Docker Containers from Images

To create a Docker container within the network, the attribute `--net` may be specified in the `run` command. Note that the run command automatically pulls an image from Docker Hub if no local images were found.


```console
# Docker run command to create a container in the network

# run in detached mode, meaning that the terminal is left free to use, requires manual shut down
docker run -d \ 

# the name of the container, specified as 'mysql'
--name mysql \

# environment variables, specific to each image
-e MYSQL_ROOT_PASSWORD=rootroot \

# Docker network in which container should run
--net mysql_network

# The image to build a container from
mysql:8.0
```

![Screenshot%202023-06-20%20at%2015.48.24.png](/assets/images/Screenshot%202023-06-20%20at%2015.48.24.png){:class="img-responsive"}

Unlike images and networks, to view docker containers, the `ps` command is used:


```console
# to show active containers
docker ps

# to show all containers, including inactive ones
docker ps -a
```

![Screenshot%202023-06-20%20at%2015.49.08.png](/assets/images/Screenshot%202023-06-20%20at%2015.49.08.png){:class="img-responsive"}

To further verify if MySQL is running in the container, the `exec` command can be combined with `-it` to start an interactive terminal running within the container. Then, we can verify if MySQL is running by typing `mysql -V`:


```console
# To open a terminal inside the container
# -it stands for interactive
# /bin/bash opens a bash terminal
docker exec -it mysql /bin/bash

# Check version of MySQL in container
mysql -V
```

![Screenshot%202023-06-19%20at%206.22.54%20PM.png](/assets/images/Screenshot%202023-06-19%20at%206.22.54%20PM.png){:class="img-responsive"}

A similar step can be performed for the phpMyAdmin container with the addition of two arguments, `--link` and `-p`:


```console
# Docker run coemmand to initialize phpmyAdmin container
docker rund -d \      # run in detached mode
--name myphpadmin \   # define container name to be phpmyadmin

# specify the link between the previously created 'mysql' container to the 'db' port within phpmyadmin
--link mysql:db \ 
--net mysql_network \ # specify docker network

# specify port on local host (8080) to port in container (80)
-p 8080:80 \
phpmyadmin:5.2        # image to build container from
```

![Screenshot%202023-06-20%20at%2015.52.39.png](/assets/images/Screenshot%202023-06-20%20at%2015.52.39.png){:class="img-responsive"}

## A05 Use phpMyAdmin to Access MySQL

Previously we defined port 8080 as the entry point in the local machine to access port 80 in the phpMyAdmin container. Thus, inputing `localhost:8080` into a web browser will bring up the phpMyAdmin page, where the username is `root` and the password was defined as `rootroot`.

![Screenshot%202023-06-20%20at%2015.53.55.png](/assets/images/Screenshot%202023-06-20%20at%2015.53.55.png){:class="img-responsive"}

After logging in, phpMyAdmin can be used to administer MySQL.

![Screenshot%202023-06-20%20at%2015.55.57.png](/assets/images/Screenshot%202023-06-20%20at%2015.55.57.png){:class="img-responsive"}

## A06 Packaging Commands into Docker Compose

The details behind each line of code within the Docker-compose YAML file was explained above and in this GitHub [repository](https://github.com/andrewyewcy/docker/blob/main/mysql.yaml). Two notable differences:
- the network is automatically defined using Docker-compose
- specifications for persistent data storage in a Docker volume are specified in the YAML file

![docker_compose_comparison.png](/assets/images/docker_compose_comparison.png){:class="img-responsive"}

# References and Acknowledgements

- phpMyAdmin logo used in thumbnail: [phpMyAdmin](https://en.wikipedia.org/wiki/PhpMyAdmin#/media/File:PhpMyAdmin_logo.svg)

- MySQL logo used in thumbnail: [MySQL](https://en.wikipedia.org/wiki/MySQL#/media/File:MySQL_logo.svg)

- Docker logo used in thumbnail: [Docker](https://en.wikipedia.org/wiki/Docker_(software)#/media/File:Docker_logo.svg)

- database icon used in setup overview diagram: <a href="https://www.flaticon.com/free-icons/database" title="database icons">Database icons created by phatplus - Flaticon</a>

- dashboard icon used in setup overview diagram: <a href="https://www.flaticon.com/free-icons/dashboard" title="dashboard icons">Dashboard icons created by Eucalyp - Flaticon</a>

- [Docker-compose for MySQL with phpMyAdmin](https://tecadmin.net/docker-compose-for-mysql-with-phpmyadmin/) by Rahul at TecAdmin.net

- [MySQL + phpMyAdmin Docker Compose](https://dev.to/devkiran/mysql-phpmyadmin-docker-compose-54h7) by Kiran Krishnan
- [MySQL Docker Hub Documentation](https://hub.docker.com/_/mysql)
- [phpMyAdmin Docker Hub Documentation](https://hub.docker.com/_/phpmyadmin)
- [Docker Tutorial for Beginners](https://www.youtube.com/watch?v=3c-iBn73dDE&ab_channel=TechWorldwithNana) by TechWorld with Nana
