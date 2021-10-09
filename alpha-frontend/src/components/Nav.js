import React,{useState,useEffect} from 'react'
import '../static/css/nav.css'

// import { faPhone } from "@fortawesome/free-solid-svg-icons";
// import { faEnvelope } from "@fortawesome/free-solid-svg-icons";
// import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";


const Nav = () => {
    const handleClick=()=>{
        document.getElementById("search").scrollIntoView();
    }
    
    return (
        <>
        <header className="header-section">
            <div className="top-section">
                
                <div class ="contacts-info">
                    <div>
                        <li className="primary-link">
                            <a href="mailto:alphaexperts245@gmail.com" className="fa fa-envelope">alphaexperts245@gmail.com</a>
                        </li>
                    </div>
                    <div>
                        <li className="primary-link">
							<a href="tel:+254718488252"><i className="fa fa-phone pr-2"></i> +254718488252</a>
                        </li>
                    </div>
                    <div className="header-social-items">
                        <ul className="header-social-list">
                            <li className="social-item"><a href="#facebook" className="facebook"><i className="fa fa-facebook"></i></a></li>
                            <li className="social-item"><a href="#twitter" className="twitter"><i className="fa fa-twitter"></i></a></li>
                            <li className="social-item"><a href="#linkedin" className="linkedin"><i className="fa fa-linkedin"></i></a></li>
                            <li className="social-item"><a href="#google" className="google"><i className="fa fa-google"></i></a></li>
                            <li className="social-item"><a href="#instagram" className="instagram"><i className="fa fa-instagram"></i></a></li>
                            <li className="social-item"><a href="#youtube" className="youtube"><i className="fa fa-youtube"></i></a></li>
                        </ul>
                    </div>
                </div>
            </div>
            <nav className="nav-bar">
                <div className="nav-logo">
                    <a href="/" className="logo">Alpha.</a>
                </div>

                <div className="nav-menu">
                    <div className="search">
                        <li class="search-icon"><a onClick={handleClick}><i class="fa fa-search"></i></a></li>
                    </div>
                    <div className="nav-options">
                        <li className="nav-item"><a href="" className="nav-link">Nairobi Securities Exchange Market Analysis</a></li>
                        {/* <li className="nav-item"><a href="#analysis" className="nav-link">Analysis</a></li>
                        <li className="nav-item"><a href="#charts" className="nav-link">Charts</a></li>
                        <li className="nav-item"><a href="#news" className="nav-link">News</a></li>
                        <li className="nav-item"><a href="#footer" className="nav-link">Contacts</a></li> */}
                    </div>
                </div>
            </nav>
        </header>
        </>
    )
}

export default Nav
