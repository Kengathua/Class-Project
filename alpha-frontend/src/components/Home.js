import {React, useEffect, useState} from 'react'

import axios from 'axios';
import '../static/css/home.css';
// import axios from 'axios';


function Home() {

    const results_url = 'http://127.0.0.1:8000/analysis/Results/';
    const [result, setResult] = useState([]);

    useEffect(()=>{
        const brand = axios.get(results_url);
        brand.then(response => {
            console.log("The results Are",response.data);
            var total = response.data.length
            setResult(response.data[total-1]);
            console.log("Output entry is:",response.data[total-1]);
            console.log("No of results is",response.data.length);
        })
        .catch(error => {
            console.error(error);
        })
    }, []);

    return (
        <>
            <section className="market-section" id="market">
                <div className="chart-container">
                    <div className="featured-chart">
                        BANK 1 CHART
                    </div>

                    <div className="featured-chart">
                        BANK 2 CHART
                    </div>

                    <div className="featured-chart">
                        BANK 3 CHART
                    </div>
                    <div className="featured-chart">
                        BANK 4 CHART
                    </div>
                    
                    <div className="featured-chart">
                        BANK 5 CHART
                    </div>
                    
                    <div className="featured-chart">
                        BANK 6 CHART
                    </div>

                </div>
            </section>
        </>
    )
}

export default Home
