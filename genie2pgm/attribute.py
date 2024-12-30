"""
@Description: Get nodes Arribution
@Author  : yxuanf
@Time    : 2023/9/1
@Site    : yxuanf@nudt.edu.cn
@File    : attribute.py 
"""

import os
import json
from genie2pgm.simplemodel import SimpleDiscreteModel


class Attributes(SimpleDiscreteModel):
    def __init__(self, xmlPath) -> None:
        super(Attributes, self).__init__(xmlPath)
        self.filename = os.path.splitext(self.path)[0].split("/")[-1]
        self.genie = self.extensions.find("genie")
        self.nodes = self.genie.findall("node")

    def getInformation(self) -> None:
        """
        @Description: Get information about the BN
        """
        information = list()
        for nodes in self.nodes:
            temp = dict()
            node = nodes.find("name")
            nodeID = nodes.attrib.get("id")
            name = node.text
            temp["id"] = nodeID
            temp["name"] = name
            temp["state"] = self.state_names[nodeID]
            information.append(temp)
        with open(f"./information/{self.filename}.json", "w", encoding="utf-8") as js:
            json.dump(information, js, ensure_ascii=False)
