import React, {Component} from "react"
import { useState } from "react";

class PostRequest extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
          ticker: '',
          duration: null,
          signal: null,
          today_date: null,
          today_price: null,
          predicted_price: null,
          actual_price: null,
          selected_date: null,
        };

    this.handleTickerChange = this.handleTickerChange.bind(this);
    this.handleDurationChange = this.handleDurationChange.bind(this);
    this.handleDateChange = this.handleDateChange.bind(this);

    this.submitTicker = this.submitTicker.bind(this);
    this.submitDuration = this.submitDuration.bind(this);
    this.submitDate = this.submitDate.bind(this);
  }

  handleTickerChange(event) {
      this.setState({ticker: event.target.value});
    };

  
  handleDurationChange(event) {
    this.setState({duration: event.target.value});
  };

  
  handleDateChange(event) {
    this.setState({selected_date: event.target.value});
  };


  submitTicker(event) {
    if (this.state.ticker){
      console.log('The selected ticker is: ' + this.state.ticker);
    }
    else{
      alert('Select the bank whose stock you want to predict')
      return
    }
    const requestOptions={
        method:"POST",
        headers:{"Content-Type":"Application/json"},
        body:JSON.stringify({
            ticker:this.state.ticker
        }),
    };
    fetch('http://127.0.0.1:8000/analysis/create-result', requestOptions)
    .then((response)=>response.json())
    .then(data => this.setState({
      signal: data.signal,
      today_date: data.today_date,
      today_price: data.today_price,
      predicted_price: data.predicted_price,
      actual_price: data.actual_price
    }));


    // event.preventDefault();
  }

  submitDuration(event) {
    if (this.state.ticker && this.state.duration){
      var duration = this.state.duration
      var regex4 = /[0-9]/g;
      // if (Number.isInteger(parseInt(duration, 10))){
      if (+duration && duration > 0){
        console.log('The selected ticker is: ' + this.state.ticker);
        console.log('The selected duration is: ' + this.state.duration);
      }
      else{
        alert('Enter a valid number of future days to be predicted')
      }
    }
    else{
      alert('Select the bank and indicate the duration in days for the stock you want to predict')
      return
    }
    
    const requestOptions={
        method:"POST",
        headers:{"Content-Type":"Application/json"},
        body:JSON.stringify({
            ticker:this.state.ticker,
            duration:this.state.duration,
        }),
    };
    fetch('http://127.0.0.1:8000/analysis/create-result', requestOptions)
    .then((response)=>response.json())
    .then(data => this.setState({
      signal: data.signal,
      today_date: data.today_date,
      today_price: data.today_price,
      predicted_price: data.predicted_price,
      actual_price: data.actual_price
    }));

  }

  submitDate(event) {
    if (this.state.ticker && this.state.selected_date){
      var starting_date = '2019-06-30'
      var selected_date = this.state.selected_date
      var date_regex  = /^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])$/
      if (date_regex.test(selected_date)){
        var v1 = new Date(starting_date)
        var v2 = new Date(selected_date)

        v1.getTime()
        v2.getTime()

        console.log(v1);
        console.log(v2);
        if (v2 <= v1) {
            alert("The Date to be predicted must be Greater than the Starting Date")
            return
        }
        else{
            console.log('The selected ticker is: ' + this.state.ticker);
            console.log('The selected date is: ' + this.state.selected_date);
        }
      }
      else{
        alert("Invalid Selected Date Format")
        return
      }
    }
    else{
      alert('Select the bank and future date of the stock you want to predict')
      return
    }

    const requestOptions={
        method:"POST",
        headers:{"Content-Type":"Application/json"},
        body:JSON.stringify({
            ticker:this.state.ticker,
            selected_date:this.state.selected_date,
        }),
    };
    fetch('http://127.0.0.1:8000/analysis/create-result', requestOptions)
    .then((response)=>response.json())
    .then(data => this.setState({
      signal: data.signal,
      today_date: data.today_date,
      today_price: data.today_price,
      predicted_price: data.predicted_price,
      actual_price: data.actual_price
    }));
  }

  render() {
    const { today_date } = this.state;
    const { today_price } = this.state;
    const { signal } = this.state;
    const { predicted_price } = this.state;
    const { actual_price } = this.state;

    return (
      <>
        <section className="analysis-section" id="analysis">
          <div className="analysis-entries">
            <div>
                <select className="ticker-selector" name="ticker" value={this.state.ticker} onChange={this.handleTickerChange}>
                    <option className="option-label" value="" disabled>--- Select Stock Name ---</option>
                    <option value="ABSA" >Absa Bank Kenya Plc</option>
                    <option value="COOP">Co-operative Bank of Kenya Ltd</option>
                    <option value="EQTY">Equity Group Holdings Plc</option>
                    <option value="HFCK">Housing Financing Company Kenya</option>
                    <option value="IMH">I&M Holdings Plc</option>
                    <option value="KCB">KCB Group Plc</option>
                    <option value="NBK">National Bank of Kenya Ltd</option>
                    <option value="NCBA">NCBA Group Plc</option>
                    <option value="SBIC">Stanbic Holdings Plc</option>
                    <option value="SCBK">	Standard Chartered Bank Kenya Ltd</option>
                </select><br/>
                <button className="submit-btn" type="submit" value="submit"  onClick={()=>this.submitTicker()}>Predict</button>
            </div>
            <div className="future-no-of-days">
                <input type="text" name="days" className="duration-input" value={this.state.duration} onChange={this.handleDurationChange} placeholder="Duration in Days (default 30 days)" /><br/>
                <button className="submit-btn" type="submit" value="submit" onClick={()=>this.submitDuration()}>Predict by Duration</button>
            </div>
            <div>
              <input type="date" id="to-date" name="to-date" className="date-input" required pattern="\d{4}-\d{2}-\d{2}" value={this.state.selected_date} onChange={this.handleDateChange} /><br/>
              <button className="submit-btn" type="submit" onClick={()=>this.submitDate()}>Predict by Specific Date</button>
            </div>
          </div>
          <div className="results">
              <div className="display-result">Current Date: <a className="display-predicted-price">{ today_date }</a></div>
              <div className="display-result">Current Price: <a className="display-actual-price">{ today_price }</a></div>
              <div className="display-result">Predicted Price: <a className="display-predicted-price">{ predicted_price }</a></div>
              <div className="display-result">Advice: <a className="display-advice">{ signal }</a></div>

              <div className="actual-price">Actual Price: <a className="display-actual-price">{ actual_price }</a></div>
            </div>
        </section>
        </>
    );
  }
}

export default PostRequest;
