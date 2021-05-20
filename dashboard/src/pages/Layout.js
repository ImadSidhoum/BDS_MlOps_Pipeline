import React, { useState } from "react"
import { Layout, Menu } from "antd"
import {
    MenuUnfoldOutlined,
    MenuFoldOutlined,
    UserOutlined,
    UploadOutlined,
    BarChartOutlined,
    ApartmentOutlined,
} from "@ant-design/icons"
import Board from "../components/Board"
import logo from "../assets/logo.png"

import {BrowserRouter, Link, Switch, Route} from "react-router-dom"
import Graph from "../components/Graph"

const { Header, Sider, Content } = Layout

const SiderDemo = () => {
    const [state, setState] = useState({
        collapsed: false,
    })

    const toggle = () => {
        setState({
            collapsed: !state.collapsed,
        })
    }
    return (
        <Layout>
            <BrowserRouter>
            <Sider trigger={null} collapsible collapsed={state.collapsed}>
                <div className="logo">
                    {state.collapsed === false ? <img src={logo} alt="ok" /> : null}
                </div>
                <Menu theme="dark" mode="inline" defaultSelectedKeys={["1"]}>
                    <Menu.Item key="1" icon={<UserOutlined />}>
                        <Link to="/">Home</Link>
                    </Menu.Item>
                    <Menu.Item key="2" icon={<BarChartOutlined />}>
                        <Link to="/outlier">Outlier</Link>
                    </Menu.Item>
                    <Menu.Item key="3" icon={<ApartmentOutlined />}>
                        <Link to="/pipeline">Pipeline</Link>
                    </Menu.Item>
                    <Menu.Item key="4" icon={<UploadOutlined />}>
                        <Link to="/contact">Contact</Link>
                    </Menu.Item>
                </Menu>
            </Sider>
            <Layout className="site-layout">
                <Header className="site-layout-background" style={{ padding: 0 }}>
                    {React.createElement(
                        state.collapsed ? MenuUnfoldOutlined : MenuFoldOutlined,
                        {
                            className: "trigger",
                            onClick: toggle,
                        },
                    )}
                </Header>
                <Content
                    className="site-layout-background"
                    style={{
                        margin: "15px 15px",
                        minHeight: 280,
                        height: "100vh",
                        overflow: "hidden",
                    }}
                >
                    <Switch>
                        <Route path="/outlier">
                            <Board />
                        </Route>
                        <Route path="/pipeline">
                            <Graph/>
                        </Route>
                        <Route path="/">
                            home
                        </Route>
                    </Switch>
                </Content>
            </Layout>
            </BrowserRouter>
        </Layout>
    )
}

export default SiderDemo
