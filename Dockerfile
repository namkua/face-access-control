
FROM jenkins/jenkins:lts

USER root

RUN apt-get update && \
    apt-get install -y \
    docker.io \
    curl \
    git \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN usermod -aG docker jenkins

RUN chown -R jenkins:jenkins /var/jenkins_home

USER jenkins
