import React from 'react';
import Board from '../components/Board';
import Menu from "../components/Menu";
import Title from '../components/Title';

const Home = () => {
    return (
        <div className="home">
            <Menu/>
            <Title/>
            <Board/>
        </div>
    );
};

export default Home;