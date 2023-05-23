import os
import time
import configparser
import pulsar
from __init__ import *

class ResultsProcess(object):
    def __init__(self, topic_name):
        self.logger = logger
        path = os.path.abspath(".")
        config_path = os.path.join(path, 'configs', 'config.ini')
        config = configparser.ConfigParser()
        config.read(config_path,encoding='utf-8')
        self.pulsar_host = config.get('Pulsar', "pulsar_host")
        # self.pulsar_port = config.get('Pulsar', "pulsar_port")
        self.pulsar_topic = topic_name
        pulsar_address = self.pulsar_host
        self.pulsar_client = pulsar.Client(pulsar_address)
        self.consumer = self.pulsar_client.subscribe(self.pulsar_topic, 
                                                     "sub-1",
                                                     consumer_type=pulsar.ConsumerType.Shared)
    
    def get_results(self):
        self.logger.info("Getting results from topic: %s...", self.pulsar_topic)
        message = self.consumer.receive()
        self.consumer.acknowledge(message)
        self.logger.info("Got results from topic: %s", self.pulsar_topic)
        # self.logger.debug("Get_Results message: %s", message)
        return message
    
    def send_message(self, message):
        self.logger.info("Sending results to topic: %s...", self.pulsar_topic)
        producer = self.pulsar_client.create_producer(self.pulsar_topic)
        producer.send(message.encode('utf-8'))
        self.logger.info("Sent results to topic: %s", self.pulsar_topic)
        self.pulsar_client.close()

    def close(self):
        self.logger.info("Closing Get_Results...")
        self.pulsar_client.close()
        self.logger.info("Closed Get_Results")