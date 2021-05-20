import React, { useState } from "react"
import { Modal, Button } from 'antd';
import { Upload, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';


const Predict = (prop) => {
    const [visible, setVisible] = useState(false)
    
    const showModal = () => {
        setVisible(true);
      };
    
    const handleCancel = () => {
        setVisible(false);
      };

    const { Dragger } = Upload;

    const props = {
        name: 'file',
        multiple: true,
        action: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
        onChange(info) {
            const { status } = info.file;
            if (status !== 'uploading') {
            console.log(info.file, info.fileList);
            }
            if (status === 'done') {
            message.success(`${info.file.name} file uploaded successfully.`);
            } else if (status === 'error') {
            message.error(`${info.file.name} file upload failed.`);
            }
        },
    };
      

    return (
        <div className="predict-container">
            <div className="predict-btn">
                <Button type="primary" onClick={()=> showModal()}>
                    Predict
                </Button>
            </div>
            <Modal
            visible={visible}
            title={"Predict: "+prop.pipeline}
            onOk={handleCancel}
            onCancel={handleCancel}
            footer={[
                <Button key="back" onClick={handleCancel}>
                Return
                </Button>,
                <Button key="submit" type="primary" onClick={handleCancel}>
                Predict
                </Button>
            ]}
            >
            <Dragger {...props}>
                <p className="ant-upload-drag-icon">
                <InboxOutlined />
                </p>
                <p className="ant-upload-text">Click or drag file to add data</p>
                {/* <p className="ant-upload-hint">
                Support for a single or bulk upload. Strictly prohibit from uploading company data or other
                band files
                </p> */}
            </Dragger>
            <p>Résult</p>
            <p>label: True</p>
            </Modal>
        </div>
    )
}

export default Predict
