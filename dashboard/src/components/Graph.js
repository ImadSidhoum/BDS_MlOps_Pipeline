import React, { useState } from 'react';
import Tree from 'react-tree-graph';
import { Drawer, Input, Button, Tooltip } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import NodeInfo from './NodeInfo';
import Predict from './Predict';


const Graph = () => {
    const [visible, setVisible] = useState(false);
    const [node, setNode] = useState({name:null});
    const [name, setName] = useState(null);
    const [port, setPort] = useState(null);
    const [addr, setAddr] = useState(null);
    const [data, setData] = useState({
        name: 'node',
        textProps: {x: -25, y: 25}
        // children: [{
        //     name: 'preprocessing 1',
        //     textProps: {x: -25, y: 25},
        //     children:[{
        //         name: 'model 1',
        //         textProps: {x: -25, y: 25}
        //     },{
        //         name: 'model 2',
        //         textProps: {x: -25, y: 25}
        //     }]
        // }, {
        //     name: 'model 3',
        //     textProps: {x: -25, y: 25}
        // }]
    });

    const showDrawer = (event, nodeKey) => {
      setVisible(true);
      setNode({name: nodeKey})
    };
  
    const onClose = () => {
      setVisible(false);
    };

    const getData = () => {

        const requestOptions = {
            mode: "cors",
            credentials: "include",
            method: "GET",
        }
        if (name && port && addr)
        {
        fetch("http://"+addr+":"+port+"/pipeline/"+name, requestOptions)
            .then(response => response.json())
            .then(e =>
                {
                    console.log(e)
                    if (e){
                        setData(e);
                    }
                }
            )
        }
    }

    
    return (
        <div className="graph">

            <div className="input-containers">
            <Input.Group compact>
                <Input style={{ width: "50%" }} placeholder="address" onChange={(e) => setAddr(e.target.value)}/>
                <Input style={{ width: "30%" }} placeholder="port" onChange={(e) => setPort(e.target.value)}/>
                <Input style={{ width: "20%" }} placeholder="name" onChange={(e) => setName(e.target.value)}/>
            </Input.Group>
            <Tooltip title="search">
                <Button type="primary" shape="circle" icon={<SearchOutlined />} onClick={()=> getData()}/>
            </Tooltip>
            <Predict pipeline={name}/>
            </div>

            <div className="custom-container">
                <Tree
                    animated={true}
                    nodeRadius={15}
                    margins={{ top: 20, bottom: 10, left: 30, right: 30 }}
                    data={data}
                    height={400}
                    width={800}
                    gProps={{onClick: showDrawer}}
                    svgProps={{
                        className: 'custom'
                    }}/>
            </div>

            <Drawer
                title={node["name"]}
                placement="right"
                closable={false}
                width={600}
                onClose={onClose}
                visible={visible}
            >
            <NodeInfo name={node["name"]} graph={data}/>
            </Drawer>
        </div>
    );
};

export default Graph;